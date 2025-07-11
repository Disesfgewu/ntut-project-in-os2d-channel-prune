import logging
import time
import math
from collections import OrderedDict
import torch
from os2d.utils import add_to_meters_in_dict, print_meters, time_since, get_trainable_parameters, checkpoint_model

class LCPFineTune:
    def __init__(self, prune_net, dataloader_train, img_normalization, box_coder, cfg, optimizer, parameters):
        """
        LCP Fine-tune 類別
        
        Args:
            prune_net: 剪枝後的網路模型
            dataloader_train: 訓練數據加載器
            img_normalization: 圖像標準化參數
            box_coder: 邊界框編碼器
            cfg: 配置對象
            optimizer: 優化器
            parameters: 可訓練參數
        """
        self._prune_net = prune_net
        self.dataloader_train = dataloader_train
        self.img_normalization = img_normalization
        self.box_coder = box_coder
        self.cfg = cfg
        self.optimizer = optimizer
        self.parameters = parameters
        
        # 基礎設置
        self._logger = logging.getLogger("LCP.FineTune")
        self._device = 'cuda' if cfg.is_cuda else 'cpu'
        self._setup_logging()
        self.ensure_model_on_device()
    
    def _setup_logging(self):
        """設置日誌記錄"""
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    # ===== 核心功能 1: 數據處理 =====
    
    def get_batch_data(self, i_batch):
        """
        從 dataloader 取得批次數據
        
        Args:
            i_batch: 批次索引
            
        Returns:
            batch_data: 批次數據
        """
        return self.dataloader_train.get_batch(i_batch)

    def prepare_batch_data_for_finetune(self, batch_data):
        """
        為 fine-tune 準備批次數據 (基於 train.py 的 prepare_batch_data)
        
        Args:
            batch_data: 原始批次數據
            
        Returns:
            processed_batch_data: 處理後的批次數據
        """
        images, class_images, loc_targets, class_targets, class_ids, class_image_sizes, \
            batch_box_inverse_transform, batch_boxes, batch_img_size = batch_data
        
        if self.cfg.is_cuda:
            images = images.cuda()
            class_images = [im.cuda() for im in class_images]
            loc_targets = loc_targets.cuda()
            class_targets = class_targets.cuda()
        
        self._logger.info(f"{images.size(0)} imgs, {len(class_images)} classes")
        
        return images, class_images, loc_targets, class_targets, class_ids, class_image_sizes, \
               batch_box_inverse_transform, batch_boxes, batch_img_size

    def shuffle_dataloader(self):
        """打亂數據加載器 (對應 train.py 的 dataloader_train.shuffle())"""
        self.dataloader_train.shuffle()

    def remap_anchor_targets(self, loc_scores, batch_img_size, class_image_sizes, batch_boxes):
        """重新映射錨點目標 (對應 train.py 的 remap_anchor_targets)"""
        return self.dataloader_train.box_coder.remap_anchor_targets(
            loc_scores, batch_img_size, class_image_sizes, batch_boxes
        )

    # ===== 核心功能 2: 單批次訓練 =====
    
    def finetune_one_batch(self, batch_data, lcp_instance, criterion, reconstruction_weight=0.1):
        """
        執行一個批次的 LCP fine-tune (基於 train.py 的 train_one_batch)
        
        Args:
            batch_data: 批次數據
            lcp_instance: LCP 實例，用於計算重建損失
            criterion: 損失函數
            reconstruction_weight: 重建損失權重
            
        Returns:
            meters: OrderedDict - 訓練指標
        """
        t_start_batch = time.time()
        
        # 設置訓練模式 (對應 train.py)
        self._prune_net.train(
            freeze_bn_in_extractor=self.cfg.train.model.freeze_bn,
            freeze_transform_params=self.cfg.train.model.freeze_transform,
            freeze_bn_transform=self.cfg.train.model.freeze_bn_transform
        )
        
        # 清零梯度 (對應 train.py)
        self.optimizer.zero_grad()
        
        # 準備批次數據
        images, class_images, loc_targets, class_targets, class_ids, class_image_sizes, \
            batch_box_inverse_transform, batch_boxes, batch_img_size = \
            self.prepare_batch_data_for_finetune(batch_data)
        
        # 前向傳播 (對應 train.py 的 net() 調用)
        loc_scores, class_scores, class_scores_transform_detached, fm_sizes, corners = \
            self._prune_net(images, class_images,
                           train_mode=True,
                           fine_tune_features=self.cfg.train.model.train_features)
        
        # 重新映射錨點目標 (對應 train.py)
        cls_targets_remapped, ious_anchor, ious_anchor_corrected = \
            self.remap_anchor_targets(loc_scores, batch_img_size, class_image_sizes, batch_boxes)
        
        # 計算 OS2D 損失 (對應 train.py 的 criterion 調用)
        os2d_losses = criterion(
            loc_scores, loc_targets,
            class_scores, class_targets,
            cls_targets_remapped=cls_targets_remapped,
            cls_preds_for_neg=class_scores_transform_detached if not self.cfg.train.model.train_transform_on_negs else None
        )
        
        # 計算 LCP 重建損失 (通過 LCP 實例)
        reconstruction_loss = 0.0
        if lcp_instance is not None:
            try:
                # 使用少量圖像計算重建損失
                reconstruction_loss = lcp_instance.compute_reconstruction_loss(
                    [0, 1, 2],  # 可配置的圖像 ID
                    layer_name='net_feature_maps.layer3.0.conv3'  # 可配置的目標層
                )
            except Exception as e:
                self._logger.warning(f"重建損失計算失敗: {e}")
                reconstruction_loss = 0.0
        
        # 組合總損失 (LCP fine-tune 策略)
        main_loss = os2d_losses["loss"]
        if torch.is_tensor(reconstruction_loss):
            total_loss = main_loss + reconstruction_weight * reconstruction_loss
        else:
            total_loss = main_loss + reconstruction_weight * torch.tensor(reconstruction_loss, device=main_loss.device)
        
        # 反向傳播 (對應 train.py)
        total_loss.backward()
        
        # 梯度裁剪 (對應 train.py)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters, self.cfg.train.optim.max_grad_norm, norm_type=2
        )
        
        # 檢查梯度 NaN (對應 train.py 的 NaN 檢查)
        if not math.isnan(grad_norm):
            self.optimizer.step()
        else:
            self._logger.warning("檢測到 NaN 梯度，跳過此次更新")
        
        # 收集訓練指標 (對應 train.py)
        meters = OrderedDict()
        for l in os2d_losses:
            meters[l] = os2d_losses[l].mean().item()
        meters["reconstruction_loss"] = reconstruction_loss.item() if torch.is_tensor(reconstruction_loss) else reconstruction_loss
        meters["total_loss"] = total_loss.item()
        meters["grad_norm"] = grad_norm
        meters["batch_time"] = time.time() - t_start_batch
        
        return meters

    # ===== 核心功能 3: 完整訓練循環 =====
    
    def finetune_training_loop(self, lcp_instance, criterion, dataloaders_eval=[], 
                              max_iters=1000, eval_interval=100, print_interval=10, 
                              reconstruction_weight=0.1):
        """
        完整的 LCP fine-tune 訓練循環 (基於 train.py 的 trainval_loop)
        
        Args:
            lcp_instance: LCP 實例，用於計算重建損失
            criterion: 損失函數
            dataloaders_eval: 評估數據加載器列表
            max_iters: 最大迭代次數
            eval_interval: 評估間隔
            print_interval: 打印間隔
            reconstruction_weight: 重建損失權重
        """
        # 初始化 (對應 train.py)
        self._logger.info("開始 LCP Fine-tune 訓練")
        t_start = time.time()
        num_steps_for_logging, meters_running = 0, {}
        
        # 確保模型在正確設備上
        self.ensure_model_on_device()
        
        # 主訓練循環 (對應 train.py 的主循環)
        i_epoch = 0
        i_batch = len(self.dataloader_train)  # 觸發新 epoch
        
        for i_iter in range(max_iters):
            # 重新開始 dataloader (對應 train.py)
            if i_batch >= len(self.dataloader_train):
                i_epoch += 1
                i_batch = 0
                self.shuffle_dataloader()
            
            # 打印迭代信息 (對應 train.py)
            if i_iter % print_interval == 0:
                self._logger.info(f"Fine-tune Iter {i_iter} ({max_iters}), epoch {i_epoch}, time {time_since(t_start)}")
            
            # 獲取訓練數據 (對應 train.py)
            t_start_loading = time.time()
            batch_data = self.get_batch_data(i_batch)
            t_data_loading = time.time() - t_start_loading
            
            i_batch += 1
            num_steps_for_logging += 1
            
            # 訓練一個批次 (對應 train.py 的 train_one_batch)
            meters = self.finetune_one_batch(batch_data, lcp_instance, criterion, reconstruction_weight)
            
            if meters is None:  # 處理錯誤情況
                continue
                
            meters["loading_time"] = t_data_loading
            
            # 打印指標 (對應 train.py)
            if i_iter % print_interval == 0:
                print_meters(meters, self._logger)
            
            # 更新運行指標 (對應 train.py)
            add_to_meters_in_dict(meters, meters_running)
            
            # 評估 (對應 train.py)
            if (i_iter + 1) % eval_interval == 0:
                self._logger.info(f"Fine-tune 評估 - 迭代 {i_iter + 1}")
                
                # 標準化指標 (對應 train.py)
                for k in meters_running:
                    meters_running[k] /= num_steps_for_logging
                
                # 打印平均指標
                print_meters(meters_running, self._logger)
                
                # 進行驗證評估
                if dataloaders_eval:
                    meters_eval = self.evaluate_finetune_model(dataloaders_eval, criterion)
                    self._logger.info(f"評估結果: {meters_eval}")
                
                # 保存檢查點
                if hasattr(self.cfg, 'output') and self.cfg.output.path:
                    self.save_finetune_checkpoint(self.cfg.output.path, i_iter)
                
                # 重置指標
                num_steps_for_logging, meters_running = 0, {}
        
        # 最終評估 (對應 train.py)
        self._logger.info("Fine-tune 最終評估")
        if dataloaders_eval:
            final_meters = self.evaluate_finetune_model(dataloaders_eval, criterion, print_per_class_results=True)
            self._logger.info(f"最終評估結果: {final_meters}")
        
        # 保存最終模型 (對應 train.py)
        if hasattr(self.cfg, 'output') and self.cfg.output.path:
            self.save_finetune_checkpoint(self.cfg.output.path, max_iters)
        
        self._logger.info(f"Fine-tune 完成，總時間: {time_since(t_start)}")

    # ===== 核心功能 4: 檢查點管理 =====
    
    def save_finetune_checkpoint(self, save_path, i_iter, extra_fields=None):
        """
        保存 fine-tune 檢查點 (基於 train.py 的 checkpoint_model)
        
        Args:
            save_path: 保存路徑
            i_iter: 當前迭代次數
            extra_fields: 額外字段
        """
        try:
            checkpoint_path = checkpoint_model(
                self._prune_net, 
                self.optimizer, 
                save_path, 
                self.cfg.is_cuda, 
                i_iter=i_iter,
                extra_fields=extra_fields
            )
            
            self._logger.info(f"Fine-tune 檢查點已保存: {checkpoint_path}")
            return checkpoint_path
        except Exception as e:
            self._logger.error(f"保存檢查點失敗: {e}")
            return None

    def load_finetune_checkpoint(self, checkpoint_path):
        """
        加載 fine-tune 檢查點
        
        Args:
            checkpoint_path: 檢查點路徑
        """
        try:
            checkpoint = torch.load(checkpoint_path)
            
            # 加載模型狀態
            self._prune_net.load_state_dict(checkpoint['model_state_dict'])
            
            # 加載優化器狀態
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self._logger.info(f"Fine-tune 檢查點已加載: {checkpoint_path}")
            return checkpoint
        except Exception as e:
            self._logger.error(f"加載檢查點失敗: {e}")
            return None

    # ===== 輔助功能 =====
    
    def evaluate_finetune_model(self, dataloaders_eval, criterion=None, print_per_class_results=False):
        """
        評估 fine-tune 模型 (基於 train.py 的 evaluate_model)
        
        Args:
            dataloaders_eval: 評估數據加載器列表
            criterion: 損失函數
            print_per_class_results: 是否打印每類結果
            
        Returns:
            meters_all: 評估結果字典
        """
        try:
            from os2d.engine.evaluate import evaluate
            
            meters_all = OrderedDict()
            for dataloader in dataloaders_eval:
                if dataloader is not None:
                    meters_val = evaluate(
                        dataloader, self._prune_net, self.cfg, 
                        criterion=criterion, 
                        print_per_class_results=print_per_class_results
                    )
                    meters_all[dataloader.get_name()] = meters_val
            
            return meters_all
        except Exception as e:
            self._logger.error(f"評估失敗: {e}")
            return {}

    def ensure_model_on_device(self):
        """確保模型在正確設備上"""
        if self.cfg.is_cuda and next(self._prune_net.parameters()).device.type != 'cuda':
            self._prune_net = self._prune_net.cuda()

    def get_current_learning_rate(self):
        """獲取當前學習率"""
        return self.optimizer.param_groups[0]['lr']

    def set_learning_rate(self, lr):
        """設置學習率"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
