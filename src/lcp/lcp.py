import torch
import torchvision.transforms as transforms
import copy 
import numpy as np
import random

from src.lcp.pruner import Pruner
from os2d.utils import get_image_size_after_resize_preserving_aspect_ratio
from src.util.detection import generate_detection_boxes
from src.util.visualize import visualize_boxes_on_image

class LCP:
    def __init__(self, net, aux_net, dataloader, img_normalization=None):
        # 預設所有模型都在 CPU 上（按需 GPU 策略）
        self._net = net.cpu()  # 原始網路
        self._prune_net = copy.deepcopy(net).cpu()  # 剪枝網路也在 CPU
        self._aux_net = aux_net
        self._dataloader = dataloader
        
        # 特徵圖快取
        self._features = {}
        self._prune_features = {}
        self._keep_dices = {}
        self._prune_net_device = 'cpu'
        self._net_device = 'cpu'
        self._pruner = Pruner(self._prune_net)
        self._prune_db = None

        # 圖像標準化參數
        if img_normalization is None:
            self.img_normalization = {
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225)
            }
        else:
            self.img_normalization = img_normalization
            
        # 圖像轉換管道
        self.transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.img_normalization["mean"], self.img_normalization["std"])
        ])
        

    def init_for_indices(self):
        """初始化 channel 保留索引"""
        layer_names = self.get_layers_name()
        for name, ch in layer_names:
            # 初始化時保留所有 channel
            self._keep_dices[name] = list(range(ch))
        print(f"[LCP] 初始化完成，共 {len(self._keep_dices)} 層的 channel 索引")

    def set_prune_db(self, prune_db):
        self._prune_db = prune_db
        self._pruner.set_prune_db(prune_db)

    def get_prune_net(self):
        return self._prune_net

    def _get_contextual_roi_align(self):
        return self._aux_net.get_contextual_roi_align()
    
    def _get_db(self):
        return self._aux_net.get_db()
    
    def get_layer_feature(self, image_tensor, layer_name='net_feature_maps'):
        """取得原始網路的 feature map - 統一 GPU 計算策略"""
        self._features = {}
        
        try:
            # 1. 確保網路在 GPU 上
            if next(self._net.parameters()).device.type != 'cuda':
                print(f"[LCP] get_layer_feature: 將 _net 移到 GPU")
                self._net = self._net.cuda()
                self._net_device = 'cuda'
                torch.cuda.empty_cache()
            
            # 2. 取得目標層
            layer = self._net
            for attr in layer_name.split('.'):
                if hasattr(layer, attr):
                    layer = getattr(layer, attr)
                elif attr.isdigit():
                    layer = layer[int(attr)]
                else:
                    raise ValueError(f"Layer {layer_name} not found in net")
            
            def hook(module, input, output):
                # 統一策略：保持在 GPU 上進行計算
                self._features[layer_name] = output
                print(f"[HOOK] {layer_name} output shape: {output.shape}")

            # 3. 執行前向傳播
            handle = layer.register_forward_hook(hook)
            self._net.eval()
            
            # 確保輸入張量在 GPU 上
            if not image_tensor.is_cuda:
                image_tensor = image_tensor.cuda()
            
            # 執行前向傳播（保持梯度追蹤能力）
            _ = self._net.net_feature_maps(image_tensor)
            
            handle.remove()
            torch.cuda.empty_cache()
            
            # 4. 返回 GPU 上的結果
            result = self._features.get(layer_name, None)
            return result
            
        finally:
            # 5. 計算完成後將網路移回 CPU
            print(f"[LCP] get_layer_feature: 將 _net 移回 CPU")
            self._net = self._net.cpu()
            self._net_device = 'cpu'
            torch.cuda.empty_cache()

    def get_prune_layer_feature(self, image_tensor, layer_name='net_feature_maps'):
        """取得剪枝網路的 feature map - 統一 GPU 計算策略"""
        self._prune_features = {}
        
        try:
            # 1. 確保剪枝網路在 GPU 上
            if next(self._prune_net.parameters()).device.type != 'cuda':
                print(f"[LCP] get_prune_layer_feature: 將 _prune_net 移到 GPU")
                self._prune_net = self._prune_net.cuda()
                self._prune_net_device = 'cuda'
                print(f"[LOG] change prune net to CUDA1")
                torch.cuda.empty_cache()
            
            # 2. 取得目標層
            layer = self._prune_net
            for attr in layer_name.split('.'):
                if hasattr(layer, attr):
                    layer = getattr(layer, attr)
                elif attr.isdigit():
                    layer = layer[int(attr)]
                else:
                    raise ValueError(f"Layer {layer_name} not found in prune_net")
            
            def hook(module, input, output):
                # 統一策略：保持在 GPU 上進行計算
                self._prune_features[layer_name] = output
                print(f"[HOOK] {layer_name} output shape: {output.shape}")

            # 3. 執行前向傳播
            handle = layer.register_forward_hook(hook)
            self._prune_net.eval()
            
            # 確保輸入張量在 GPU 上
            if not image_tensor.is_cuda:
                image_tensor = image_tensor.cuda()
            
            # 執行前向傳播（保持梯度追蹤能力）
            _ = self._prune_net.net_feature_maps(image_tensor)
            
            handle.remove()
            torch.cuda.empty_cache()
            
            # 4. 返回 GPU 上的結果
            result = self._prune_features.get(layer_name, None)
            return result
            
        finally:
            # 5. 計算完成後將剪枝網路移回 CPU
            # print(f"[LCP] get_prune_layer_feature: 將 _prune_net 移回 CPU")
            # self._prune_net = self._prune_net.cpu()
            # self._prune_net_device = 'cpu'
            # print(f"[LOG] change prune net to CPU1")
            torch.cuda.empty_cache()

    def get_image_tensor_from_dataloader(self, image_id, resize_target=1500, is_cuda=False):
        """
        從 dataloader 取得 PIL 圖片，resize、轉 tensor、標準化，回傳 [1, C, H, W] tensor
        按需 GPU 策略：根據當前計算需求決定設備位置
        """
        try:
            # 1. 取得 PIL Image
            img = self._dataloader._get_dataset_image_by_id(image_id)
            
            # 2. resize（保持比例，對應 detection.py 實作）
            from os2d.utils import get_image_size_after_resize_preserving_aspect_ratio
            h, w = get_image_size_after_resize_preserving_aspect_ratio(
                h=img.size[1], w=img.size[0], target_size=resize_target
            )
            img = img.resize((w, h))
            
            # 3. 轉 tensor + 標準化
            img_tensor = self.transform_image(img)  # [C, H, W]
            img_tensor = img_tensor.unsqueeze(0)    # [1, C, H, W]
            
            # 4. 智能設備管理：根據計算需求決定設備位置
            img_tensor = img_tensor.cuda()
            # print(f"[LCP] Image tensor moved to GPU - shape: {img_tensor.shape}")
            return img_tensor
            
        except Exception as e:
            print(f"[ERROR] Failed to process image {image_id}: {e}")
            raise e

    def get_layers_name(self):
        """
        列出 backbone（net_feature_maps）所有層的名稱與 channel 數量
        Returns:
            layers_info: list of (name, channel數) tuple
        """
        layers_info = []
        def traverse(module, prefix=''):
            for name, submodule in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                # Conv/BN 層
                if hasattr(submodule, 'out_channels'):
                    layers_info.append((full_name, submodule.out_channels))
                # 線性層
                elif hasattr(submodule, 'out_features'):
                    layers_info.append((full_name, submodule.out_features))
                # 遞迴
                traverse(submodule, full_name)
        # 只列出 backbone
        traverse(self._prune_net.net_feature_maps)
        return layers_info
    
    # def compute_reconstruction_loss(self, image_ids, layer_name):
    #     """
    #     計算多張圖像在指定層的重建誤差（均方誤差），對齊 LCP 論文 3.3 節設計。
    #     Args:
    #         image_ids: list of image_id
    #         layer_name: 目標層名稱（如 'net_feature_maps.layer3.2.conv2'）
    #     Returns:
    #         float: 多圖平均的重建誤差
    #     """
    #     is_cuda = next(self._net.parameters()).is_cuda
    #     losses = []
    #     for image_id in image_ids:
    #         img_tensor = self.get_image_tensor_from_dataloader(image_id, is_cuda=is_cuda)
    #         feature_map_orig = self.get_layer_feature(img_tensor, layer_name=layer_name)
    #         feature_map_pruned = self.get_prune_layer_feature(img_tensor, layer_name=layer_name)
    #         if feature_map_orig is None or feature_map_pruned is None:
    #             raise ValueError(f"Feature map not found for layer {layer_name} and image_id {image_id}")
    #         if feature_map_orig.shape != feature_map_pruned.shape:
    #             raise ValueError(f"Feature map shape mismatch: {feature_map_orig.shape} vs {feature_map_pruned.shape}")
    #         Q = feature_map_orig.numel()
    #         loss = 0.5 / Q * torch.norm(feature_map_orig - feature_map_pruned, p=2) ** 2
    #         print( f"[LOG] Image ID: {image_id}, Layer: {layer_name}, Loss: {loss.item()}, Q: {Q}, torch.norm(feature_map_orig - feature_map_pruned, p=2) = {torch.norm(feature_map_orig - feature_map_pruned, p=2)}")
    #         losses.append(loss.item())
    #     return float(np.mean(losses))

    def compute_reconstruction_loss(self, image_ids, layer_name, keep_out_idx=None):
        """
        計算多張圖像在指定層的重建誤差（均方誤差），支援 channel slicing。
        計算完成後才把資料移出 GPU 策略
        """
        losses = []
        
        try:
            for image_id in image_ids:
                # 清理記憶體
                torch.cuda.empty_cache()
                
                # 使用 GPU 處理圖像張量
                img_tensor = self.get_image_tensor_from_dataloader(image_id, is_cuda=True)
                
                # 提取特徵圖（保持在 GPU 上）
                feature_map_orig = self.get_layer_feature(img_tensor, layer_name=layer_name)
                feature_map_pruned = self.get_prune_layer_feature(img_tensor, layer_name=layer_name)
                
                # Channel slicing（如果需要）
                if keep_out_idx is not None:
                    feature_map_orig = feature_map_orig[:, keep_out_idx, :, :]
                    
                # 錯誤檢查
                if feature_map_orig is None or feature_map_pruned is None:
                    raise ValueError(f"Feature map not found for layer {layer_name} and image_id {image_id}")
                    
                if feature_map_orig.shape != feature_map_pruned.shape:
                    raise ValueError(f"Feature map shape mismatch: {feature_map_orig.shape} vs {feature_map_pruned.shape}")
                
                # 計算重建誤差（在 GPU 上）
                Q = feature_map_orig.numel()
                loss = 0.5 / Q * torch.norm(feature_map_orig - feature_map_pruned, p=2) ** 2
                
                print(f"[LOG] Image ID: {image_id}, Layer: {layer_name}, Loss: {loss.item()}, Q: {Q}")
                
                # 保持 tensor 格式在 GPU 上，不立即移出
                losses.append(loss)
                
                # 清理中間變數但保持 losses 在 GPU
                del img_tensor, feature_map_orig, feature_map_pruned
                torch.cuda.empty_cache()
            
            # 在 CPU 上計算最終結果
            final_loss = torch.stack(losses).mean().cpu()
            print(f"[LOG] 所有計算完成，最終 loss: {final_loss.item()}")
            
            return final_loss
            
        finally:
            # 計算完成後將網路移回 CPU
            print(f"[LCP] compute_reconstruction_loss: 計算完成，將網路移回 CPU")
            torch.cuda.empty_cache()

    def compute_joint_loss(self, layer_name, lambda_rate=1.0, use_image_num=None, random_seed=None):
        """
        隨機抽取 image_ids，計算 joint loss（重建誤差 + λ × auxiliary loss），完全用現有 API。
        Args:
            layer_name: backbone 層名稱
            lambda_rate: joint loss 中 auxiliary loss 的權重
            use_image_num: 隨機抽取多少張圖（int），若為 None 則用全部
            random_seed: 隨機種子
        Returns:
            joint_loss: torch scalar
            rec_loss_mean: float
            aux_loss_mean: float
        """
        image_id_all = list(map(int, self._get_db().get_image_ids()))
        if use_image_num is not None and use_image_num < len(image_id_all):
            if random_seed is not None:
                random.seed(random_seed)
            image_id_all = random.sample(image_id_all, use_image_num)

        # 1. 用你的 API 計算重建誤差（多圖平均）
        rec_loss_mean = self.compute_reconstruction_loss(image_id_all, layer_name, keep_out_idx=self._keep_dices.get(layer_name, None) )
        # 2. 用 aux_net.aux_loss 計算所有 box 的 auxiliary loss
        aux_losses = []
        for image_id in image_id_all:
            class_ids = list(map(int, self._get_db().get_class_ids_by_image_id(image_id).keys()))
            for class_id in class_ids:
                point1_xs = self._get_db().get_specific_data(image_id, class_id, "point1_x")
                point1_ys = self._get_db().get_specific_data(image_id, class_id, "point1_y")
                point2_xs = self._get_db().get_specific_data(image_id, class_id, "point2_x")
                point2_ys = self._get_db().get_specific_data(image_id, class_id, "point2_y")
                if len(point1_xs) == 0 or len(point1_ys) == 0 or len(point2_xs) == 0 or len(point2_ys) == 0:
                    continue
                # 這裡只取第一組 box（如需全遍歷可再加迴圈）
                point1 = (point1_xs[0], point1_ys[0])
                point2 = (point2_xs[0], point2_ys[0])
                aux_loss = self._aux_net.aux_loss(image_id, class_id, point1, point2)
                aux_losses.append(aux_loss if torch.is_tensor(aux_loss) else torch.tensor(aux_loss))

        if len(aux_losses) == 0:
            raise ValueError("No valid auxiliary losses computed, check your data and db queries.")

        aux_loss_mean = torch.stack(aux_losses).mean()
        # 3. 組合 joint loss
        joint_loss = rec_loss_mean + lambda_rate * aux_loss_mean
        return joint_loss, rec_loss_mean, aux_loss_mean

    def compute_channel_importance(self, layer_name, lambda_rate=1.0, use_image_num=None, random_seed=None):
      """計算 channel importance - 確保設備一致性"""
      
      # 1. 確保 _prune_net 在 GPU 上進行梯度計算
      if next(self._prune_net.parameters()).device.type != 'cuda':
          print(f"[LCP] compute_channel_importance: 將 _prune_net 移到 GPU")
          self._prune_net = self._prune_net.cuda()
          self._prune_net_device = 'cuda'
          print(f"[LOG] change prune net to CUDA2")
      
      # 2. 取得目標層
      layer = self._prune_net
      for attr in layer_name.split('.'):
          if hasattr(layer, attr):
              layer = getattr(layer, attr)
          elif attr.isdigit():
              layer = layer[int(attr)]
      
      layer.weight.requires_grad_(True)
      
      # 3. 計算 joint loss
      self._aux_net._context_roi_align.backprop = True
      joint_loss, rec_loss_mean, aux_loss_mean = self.compute_joint_loss(
          layer_name, lambda_rate, use_image_num, random_seed
      )
      self._aux_net._context_roi_align.backprop = False
      
      # 4. 確保 joint_loss 在 GPU 上進行梯度計算
      if not joint_loss.is_cuda:
          joint_loss = joint_loss.cuda()
      
      print(f"[LOG] Computing channel importance for {layer_name}")
      print(f"[ LOG] Joint loss: {joint_loss:.6f}")
      
      # 5. GPU 上的梯度計算（現在設備一致）
      print( f"[LOG] device for prune net : {self._prune_net_device}")
      print( f"[LOG] device for net : {self._net_device}")
      
      self._prune_net.zero_grad()
      joint_loss.backward()
      
      grad = layer.weight.grad
      if grad is None:
          raise ValueError(f"No gradient computed for layer {layer_name}")
      
      importance = grad.pow(2).sum(dim=(1, 2, 3)).detach().cpu().numpy()
      
      print(f"[LOG] Channel importance 計算完成")
      print(f"[LCP] compute_channel_importance: 將 _prune_net 移回 CPU")
      self._prune_net = self._prune_net.cpu()
      self._prune_net_device = 'cpu'
      print(f"[LOG] change prune net to CPU3")
      torch.cuda.empty_cache()
      return importance
         
    def _get_layer_by_name(self, layer_name, net):
        """
        根據層名稱獲取層對象
        
        Args:
            layer_name: 層名稱（支持點分割路徑）
            
        Returns:
            torch.nn.Module: 層對象，如果找不到則返回 None
        """
        try:
            # 移除 'net_feature_maps.' 前綴（如果存在）
            if layer_name.startswith('net_feature_maps.'):
                layer_path = layer_name.replace('net_feature_maps.', '')
            else:
                layer_path = layer_name
            
            # 從網路開始遍歷
            layer = net.net_feature_maps
            
            # 按路徑逐級獲取層對象
            for attr in layer_path.split('.'):
                if hasattr(layer, attr):
                    layer = getattr(layer, attr)
                elif attr.isdigit():
                    layer = layer[int(attr)]
                else:
                    print(f"[ERROR] 無法找到屬性: {attr} in {layer_path}")
                    return None
            
            return layer
            
        except Exception as e:
            print(f"[ERROR] 獲取層 {layer_name} 失敗: {e}")
            return None

    def compute_channel_importance_no_grad(self, layer_name, lambda_rate=1.0, use_image_num=None, random_seed=None):
        """
        基於數學推導的無梯度通道重要性計算
        實現統合公式：S_k = w1×L1 + w2×Var + w3×MeanDev + w4×Energy + w5×Sparsity
        """
        print(f"[LCP] 開始基於數學推導的無梯度通道重要性計算 - {layer_name}")
        
        self._net = self._net.cuda()
        self._prune_net = self._prune_net.cuda()
        # 準備圖像數據
        image_id_all = list(map(int, self._get_db().get_image_ids()))
        if use_image_num is not None and use_image_num < len(image_id_all):
            if random_seed is not None:
                random.seed(random_seed)
            image_id_all = random.sample(image_id_all, use_image_num)
        
        # 收集特徵統計和損失近似
        all_feature_stats = []
        all_reconstruction_errors = []
        all_auxiliary_losses = []
        print( image_id_all )
        for image_id in image_id_all:
            # 1. 特徵統計計算（基於理論公式）
            
            feature_stats = self._extract_and_compute_theoretical_statistics(image_id, layer_name)
            print( image_id , feature_stats )
            if feature_stats:
                all_feature_stats.append(feature_stats)
            
            # 2. 重建誤差近似
            reconstruction_error = self._approximate_reconstruction_error_with_theory(image_id, layer_name)
            all_reconstruction_errors.append(reconstruction_error)
            
            # 3. 輔助損失近似
            auxiliary_loss = self._approximate_auxiliary_loss_with_theory(image_id)
            all_auxiliary_losses.append(auxiliary_loss)
        
        # 4. 基於理論公式整合最終重要性
        importance_scores = self._integrate_importance_with_mathematical_formula(
            all_feature_stats, all_reconstruction_errors, all_auxiliary_losses, lambda_rate
        )
        
        print(f"[LCP] 基於數學推導的無梯度計算完成")
        return importance_scores

    def _extract_layer_features_no_grad(self, network, img_tensor, layer_name):
        """
        無梯度特徵提取
        從指定網路層提取特徵圖
        """
        features = {}
        
        # 獲取目標層
        target_layer = network
        for attr in layer_name.split('.'):
            if hasattr(target_layer, attr):
                target_layer = getattr(target_layer, attr)
            elif attr.isdigit():
                target_layer = target_layer[int(attr)]
            else:
                print(f"[ERROR] 無法找到層: {attr} in {layer_name}")
                return None
        
        def hook(module, input, output):
            features[layer_name] = output.detach().cpu()
        
        # 註冊 hook
        handle = target_layer.register_forward_hook(hook)
        
        try:
            with torch.no_grad():
                network.eval()
                # 確保輸入在正確的設備上
                if hasattr(network, 'net_feature_maps'):
                    _ = network.net_feature_maps(img_tensor)
                else:
                    _ = network(img_tensor)
        finally:
            handle.remove()
        
        return features.get(layer_name, None)

    def _approximate_auxiliary_loss_with_theory(self, image_id):
        """
        基於理論的輔助損失近似計算
        """
        try:
            # 獲取該圖像的所有類別
            class_ids = list(map(int, self._get_db().get_class_ids_by_image_id(image_id).keys()))
            
            if not class_ids:
                return 0.0
            
            total_aux_loss = 0.0
            count = 0
            
            for class_id in class_ids:
                # 獲取邊界框資料
                point1_xs = self._get_db().get_specific_data(image_id, class_id, "point1_x")
                point1_ys = self._get_db().get_specific_data(image_id, class_id, "point1_y")
                point2_xs = self._get_db().get_specific_data(image_id, class_id, "point2_x")
                point2_ys = self._get_db().get_specific_data(image_id, class_id, "point2_y")
                
                if len(point1_xs) == 0:
                    continue
                
                # 使用第一組邊界框
                point1 = (float(point1_xs[0]), float(point1_ys[0]))
                point2 = (float(point2_xs[0]), float(point2_ys[0]))
                
                # 計算輔助損失近似
                if hasattr(self._aux_net, 'approximate_ar_loss_no_grad') and hasattr(self._aux_net, 'approximate_ac_loss_no_grad'):
                    ar_loss = self._aux_net.approximate_ar_loss_no_grad(image_id, class_id, point1, point2)
                    ac_loss = self._aux_net.approximate_ac_loss_no_grad(image_id, class_id, point1, point2)
                    aux_loss = ar_loss + ac_loss
                else:
                    # 降級到簡單估計
                    aux_loss = 1.0  # 預設值
                
                total_aux_loss += aux_loss
                count += 1
            
            return total_aux_loss / max(count, 1)
            
        except Exception as e:
            print(f"[WARNING] 輔助損失近似計算失敗: {e}")
            return 0.0

    def _approximate_reconstruction_error_with_theory(self, image_id, layer_name):
        """
        基於理論的重建誤差近似計算
        """
        try:
            # 獲取圖像張量
            img_tensor = self.get_image_tensor_from_dataloader(image_id, is_cuda=False)
            
            with torch.no_grad():
                # 提取原始網路特徵
                orig_features = self._extract_layer_features_no_grad(self._net, img_tensor, layer_name)
                # 提取剪枝網路特徵
                pruned_features = self._extract_layer_features_no_grad(self._prune_net, img_tensor, layer_name)
            
            if orig_features is None or pruned_features is None:
                return 0.0
            
            # 計算重建誤差（MSE）
            if orig_features.shape != pruned_features.shape:
                print(f"[WARNING] 特徵維度不匹配: {orig_features.shape} vs {pruned_features.shape}")
                return 0.0
            
            mse = torch.mean((orig_features - pruned_features) ** 2)
            return mse.item()
            
        except Exception as e:
            print(f"[WARNING] 重建誤差計算失敗: {e}")
            return 0.0


    def _extract_and_compute_theoretical_statistics(self, image_id, layer_name):
        """基於理論公式提取和計算特徵統計"""
        try:
            # 無梯度特徵提取
            img_tensor = self.get_image_tensor_from_dataloader(image_id, is_cuda=False)
            
            with torch.no_grad():
                orig_features = self._extract_layer_features_no_grad(self._net, img_tensor, layer_name)
                print( "[LOG] 原始網路特徵提取完成")
                pruned_features = self._extract_layer_features_no_grad(self._prune_net, img_tensor, layer_name)
                print( "[LOG] 剪枝網路特徵提取完成")
            if orig_features is None:
                return None
            
            # 實現理論報告中的完整統計公式
            return self._compute_mathematical_feature_statistics(orig_features, pruned_features)
            
        except Exception as e:
            print(f"[WARNING] 理論統計計算失敗: {e}")
            return None

    def _compute_mathematical_feature_statistics(self, orig_features, pruned_features):
        """實現理論推導中的數學統計公式"""
        if orig_features.dim() == 4:
            orig_features = orig_features.squeeze(0)  # [C, H, W]
        
        C, H, W = orig_features.shape
        
        # 全域統計（理論基礎）
        global_mean = torch.mean(orig_features)
        global_std = torch.std(orig_features) + 1e-8
        
        # 逐通道統計（實現理論公式）
        channel_stats = []
        for k in range(C):
            channel = orig_features[k]  # [H, W]
            
            # 理論公式中的各項統計指標
            l1_norm = torch.sum(torch.abs(channel)) / (H * W)
            variance = torch.var(channel)
            mean_val = torch.mean(channel)
            mean_deviation = torch.abs(mean_val - global_mean)
            energy = torch.sum(channel ** 2) / (H * W)
            sparsity = torch.sum(torch.abs(channel) < 1e-6).float() / (H * W)
            
            # 實現理論報告中的統合公式
            # S_k = w1×L1/σ + w2×Var/σ² + w3×|μ-μ_g|/σ + w4×E/σ² + w5×(1-sparsity)
            importance_score = (
                0.3 * (l1_norm / global_std) +                    # 重建誤差項
                0.2 * (variance / (global_std ** 2)) +            # 分類損失項
                0.2 * (mean_deviation / global_std) +             # 回歸損失項
                0.15 * (energy / (global_std ** 2)) +             # 能量項
                0.15 * (1.0 - sparsity)                           # 反稀疏性項
            )
            
            channel_stats.append({
                'l1_norm': l1_norm.item(),
                'variance': variance.item(),
                'mean_deviation': mean_deviation.item(),
                'energy': energy.item(),
                'sparsity': sparsity.item(),
                'importance': importance_score.item()
            })
        
        return {
            'channels': channel_stats,
            'global_mean': global_mean.item(),
            'global_std': global_std.item()
        }

    def _integrate_importance_with_mathematical_formula(self, feature_stats, reconstruction_errors, auxiliary_losses, lambda_rate):
        """基於數學推導整合最終重要性分數"""
        if not feature_stats:
            raise ValueError("無有效的特徵統計資料")
        
        num_channels = len(feature_stats[0]['channels'])
        importance_scores = np.zeros(num_channels)
        
        for i in range(num_channels):
            # 收集該通道的統計重要性（來自理論公式）
            channel_importances = []
            for stats in feature_stats:
                if stats and 'channels' in stats:
                    channel_importances.append(stats['channels'][i]['importance'])
            
            if channel_importances:
                # 基礎統計重要性（理論公式核心）
                base_importance = np.mean(channel_importances)
                
                # 重建誤差貢獻（理論公式中的 L_re 項）
                reconstruction_contribution = np.mean(reconstruction_errors) * 0.1
                
                # 輔助損失貢獻（理論公式中的 L_a 項）
                auxiliary_contribution = np.mean(auxiliary_losses) * lambda_rate * 0.1
                
                # 最終重要性（實現聯合損失公式：L = L_re + α×L_a）
                final_importance = base_importance + reconstruction_contribution + auxiliary_contribution
                
                importance_scores[i] = final_importance
        
        return importance_scores
    
    def get_channel_selection_by_no_grad(
        self,
        layer_name: str,
        discard_rate: float = 0.5,
        lambda_rate: float = 1.0,
        use_image_num: int = None,
        random_seed: int = None):
        """
        基於無梯度重要性分數的通道選擇
        Args
        ----
        layer_name   : 目標層名稱 (如 'net_feature_maps.layer3.2.conv2')
        discard_rate : 捨棄比例 (0~1)；0.5 代表剪掉一半通道
        lambda_rate  : joint-loss 中 auxiliary loss 的權重
        use_image_num: 隨機抽取多少張圖像計算重要性；None 則全部
        random_seed  : 隨機種子（控制抽樣與同分時的隨機排序）
        Returns
        -------
        keep_idx    : `np.ndarray`，保留通道索引 (升冪)
        discard_idx : `np.ndarray`，捨棄通道索引 (升冪)
        """
        # -------- 0. 參數檢查 --------
        if not (0.0 <= discard_rate < 1.0):
            raise ValueError("discard_rate 必須位於 [0, 1) 區間")

        if random_seed is not None:
            np.random.seed(random_seed)

        # -------- 1. 計算通道重要性 --------
        importance = self.compute_channel_importance_no_grad(
            layer_name   = layer_name,
            lambda_rate  = lambda_rate,
            use_image_num= use_image_num,
            random_seed  = random_seed
        )  # numpy array, shape = [C]

        C = importance.shape[0]
        if C == 0:
            raise ValueError(f"{layer_name} 無有效通道")

        # -------- 2. 排序（由小到大）--------
        sorted_idx = np.argsort(importance)           # 重要性小 ➜ 排前面
        n_discard  = int(np.floor(C * discard_rate))  # 需捨棄的通道數

        # 若 discard_rate 很小可能為 0；確保至少保留 1 個通道
        n_discard = min(max(n_discard, 0), C - 1)

        discard_idx = sorted_idx[:n_discard]
        keep_idx    = sorted_idx[n_discard:]          # 其餘通道保留

        # -------- 3. 同步內部索引表 --------
        self._keep_dices[layer_name] = keep_idx.tolist()

        # -------- 4. 回傳結果 --------
        return np.sort(keep_idx), np.sort(discard_idx)

    def get_channel_selection_and_write_to_db( self, layer_name , discard_rate=0.5, use_image_num=3, random_seed=42):
        """
        layer_name : ex 'layer3.2.conv2'
        """
        keep, discard = self.get_channel_selection_by_no_grad(
            layer_name   = f"net_feature_maps.{layer_name}",
            discard_rate = discard_rate,
            lambda_rate  = 1.0,
            use_image_num= use_image_num,
            random_seed  = random_seed
        )

        print(f"layer {layer_name} , 預計保留通道數量: {len(keep)}/{len(keep)+len(discard)}, 預計捨棄通道數量: {len(discard)}/{len(keep)+len(discard)}")

        self._prune_db.write_data(
            layer = f"net_feature_maps.{layer_name}",
            original_channel_num= len(keep) + len(discard),
            num_of_keep_channel = len(keep),
            keep_index  = keep
        )

    def prune_layer(self, layer_name, discard_rate=0.5, use_image_num=3, random_seed=42):
        self.get_channel_selection_and_write_to_db(
            layer_name   = layer_name,
            discard_rate = discard_rate,
            use_image_num= use_image_num,
            random_seed  = random_seed
        )

        self._pruner.prune_layer(
            layer_name   = layer_name
        )

    def debug_for_test_vision(self, dataloader_train, img_normalization, box_coder, cfg, count=1):
        image_id = 0
        class_id = 0
        self._prune_net = self._prune_net.cuda()
        get, labels, scores = generate_detection_boxes(dataloader_train, self._prune_net, img_normalization, box_coder, image_id, class_id, cfg, class_num=count*2)
        from os2d.modeling.box_coder import BoxList

        original_image = dataloader_train._get_dataset_image_by_id(image_id)
        orig_h, orig_w = original_image.size[1], original_image.size[0]
        
        image_height, image_width = get_image_size_after_resize_preserving_aspect_ratio(h=original_image.size[1],
                                                                                        w=original_image.size[0],
                                                                                        target_size=1500)
        box_list = BoxList(get, (image_width, image_height), mode="xyxy")
        box_list.add_field("labels", labels)
        box_list.add_field("scores", scores)  # Add scores field for proper visualization
        visualize_boxes_on_image(
            image_id=image_id,
            boxes_one_image=box_list,
            dataloader=dataloader_train,
            cfg=cfg,
            class_ids=class_id,
            path="detection",
            is_detection=True,
            showfig=True,  # Specify this is detection visualization
        )