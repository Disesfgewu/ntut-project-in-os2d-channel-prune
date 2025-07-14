from os2d.engine.objective import Os2dObjective
import torch
import torch.nn as nn
import numpy as np

class LCPFinetuneCritserion(nn.Module):
    def __init__(self, original_criterion, aux_net, auxiliary_weight=1.0):
        """
        使用組合模式包裝 Os2dObjective
        
        Args:
            original_criterion: Os2dObjective 實例
            aux_net: 輔助網路實例
            auxiliary_weight: 輔助損失權重
        """
        super(LCPFinetuneCriterion, self).__init__()
        self.original_criterion = original_criterion
        self.aux_net = aux_net
        self.auxiliary_weight = auxiliary_weight
    
    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets,
            cls_targets_remapped=None, cls_preds_for_neg=None,
            patch_mining_mode=False, batch_idx=None):
        """委託給原始 criterion 並加入輔助損失"""
        
        # 委託給原始 Os2dObjective
        original_result = self.original_criterion(
            loc_preds, loc_targets, cls_preds, cls_targets,
            cls_targets_remapped, cls_preds_for_neg, patch_mining_mode
        )
        
        # 處理返回值
        if patch_mining_mode:
            original_losses, losses_per_anchor = original_result
        else:
            original_losses = original_result
        
        # 計算並加入輔助損失 - 使用 batch_idx 而不是 batch_data
        auxiliary_loss = self.compute_auxiliary_loss_from_batch(batch_idx)
        total_loss = original_losses["loss"] + self.auxiliary_weight * auxiliary_loss
        
        print(f"[LCP Loss] Original loss: {original_losses['loss']:.6f}")
        print(f"[LCP Loss] Auxiliary loss: {auxiliary_loss:.6f}")
        print(f"[LCP Loss] Auxiliary weight: {self.auxiliary_weight}")
        print(f"[LCP Loss] Total loss: {total_loss:.6f}")
        
        # 更新損失字典
        lcp_losses = original_losses.copy()
        lcp_losses["loss"] = total_loss
        lcp_losses["auxiliary_loss"] = auxiliary_loss
        lcp_losses["original_loss"] = original_losses["loss"]
        
        if patch_mining_mode:
            return lcp_losses, losses_per_anchor
        else:
            return lcp_losses


    def get_image_ids_for_batch_index(self, batch_idx):
        """獲取指定批次索引的圖像 ID"""
        return self.aux_net.get_contextual_roi_align().dataloader.get_image_ids_for_batch_index(batch_idx)

    def get_dataloader(self):
        """獲取數據加載器"""
        return self.aux_net.get_contextual_roi_align().dataloader

    def get_database(self):
        return self.aux_net.get_db()

    def compute_auxiliary_loss_from_batch(self, batch_idx):
        """從批次數據計算輔助損失 - 限制樣本數量版本"""
        print(f"[LCP Debug] Computing auxiliary loss...")
        print(f"[LCP Debug] batch_data is None: {batch_idx is None}")
        print(f"[LCP Debug] aux_net is None: {self.aux_net is None}")

        if batch_idx is None or self.aux_net is None:
            print(f"[LCP Debug] Returning zero auxiliary loss")
            return torch.tensor(0.0, requires_grad=True)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        total_aux_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        try:
            image_ids = self.get_image_ids_for_batch_index(batch_idx)
            batch = self.get_dataloader().get_batch(batch_idx)
            images, class_images, loc_targets, cls_targets, class_ids, class_sizes, transforms, boxes, img_sizes = batch

            # 調試：檢查 class_ids 的結構
            print(f"[LCP Debug] class_ids type: {type(class_ids)}")
            print(f"[LCP Debug] class_ids shape/length: {len(class_ids) if hasattr(class_ids, '__len__') else 'No length'}")
            if len(class_ids) > 0:
                print(f"[LCP Debug] class_ids[0] type: {type(class_ids[0])}")

            db = self.get_database()
            count = 0
            valid_samples = 0
            
            # 收集所有可能的樣本
            all_samples = []
            
            for i, image_id in enumerate(image_ids):
                # 修正：處理 class_ids 的不同結構
                if isinstance(class_ids[i], (list, tuple, np.ndarray)):
                    current_class_ids = class_ids[i]
                else:
                    current_class_ids = [class_ids[i]]
                
                for j, class_id in enumerate(current_class_ids):
                    class_id = int(class_id)
                    
                    try:
                        point1_xs = db.get_specific_data(image_id, class_id, 'point1_x')
                        point1_ys = db.get_specific_data(image_id, class_id, 'point1_y')
                        point2_xs = db.get_specific_data(image_id, class_id, 'point2_x')
                        point2_ys = db.get_specific_data(image_id, class_id, 'point2_y')

                        # 檢查數據完整性
                        if not (point1_xs and point1_ys and point2_xs and point2_ys):
                            continue
                        
                        # 確保所有列表長度一致
                        min_length = min(len(point1_xs), len(point1_ys), len(point2_xs), len(point2_ys))
                        
                        for k in range(min_length):
                            try:
                                point1 = (float(point1_xs[k]), float(point1_ys[k]))
                                point2 = (float(point2_xs[k]), float(point2_ys[k]))
                                
                                # 收集樣本而不是立即計算
                                all_samples.append({
                                    'image_id': image_id,
                                    'class_id': class_id,
                                    'point1': point1,
                                    'point2': point2
                                })
                                
                            except (ValueError, TypeError) as e:
                                print(f"[LCP Debug] Error processing sample {k} for image_id={image_id}, class_id={class_id}: {e}")
                                continue
                                
                    except Exception as e:
                        print(f"[LCP Debug] Error accessing data for image_id={image_id}, class_id={class_id}: {e}")
                        continue
            
            # 計算樣本限制：總數的5%，最多10個
            total_samples = len(all_samples)
            max_samples = min(max(1, int(total_samples * 0.05)), 10)
            
            print(f"[LCP Debug] Total available samples: {total_samples}")
            print(f"[LCP Debug] Using samples: {max_samples} (5% of total, max 10)")
            
            if total_samples == 0:
                print(f"[LCP Debug] No valid samples found, returning zero loss")
                return torch.tensor(0.0, device=device, requires_grad=True)
            
            # 隨機選擇樣本或取前N個
            import random
            if total_samples > max_samples:
                selected_samples = random.sample(all_samples, max_samples)
            else:
                selected_samples = all_samples
            
            # 計算選中樣本的輔助損失
            for sample in selected_samples:
                try:
                    sample_aux_loss = self.aux_net.aux_loss(
                        sample['image_id'], 
                        sample['class_id'], 
                        sample['point1'], 
                        sample['point2']
                    )
                    
                    # 確保損失在正確設備上
                    if sample_aux_loss.device != device:
                        sample_aux_loss = sample_aux_loss.to(device)
                    
                    total_aux_loss = total_aux_loss + sample_aux_loss
                    count += 1
                    
                    # 檢查是否為有效損失值
                    if sample_aux_loss.item() > 0:
                        valid_samples += 1
                        
                except Exception as e:
                    print(f"[LCP Debug] Error computing aux_loss for sample: {e}")
                    continue
            
            # 計算平均輔助損失
            if count > 0:
                average_aux_loss = total_aux_loss / count
                print(f"[LCP Debug] Total samples processed: {count}/{max_samples}")
                print(f"[LCP Debug] Valid samples (loss > 0): {valid_samples}")
                print(f"[LCP Debug] Average auxiliary loss: {average_aux_loss.item():.6f}")
                return average_aux_loss
            else:
                print(f"[LCP Debug] No valid samples processed, returning zero loss")
                return torch.tensor(0.0, device=device, requires_grad=True)
                
        except Exception as e:
            print(f"[LCP Error] Failed to compute auxiliary loss: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(0.0, device=device, requires_grad=True)

    def extract_samples_from_batch_alternative(self, batch_data):
        """
        替代的提取方法，適用於不同的數據格式
        """
        samples = []
        
        if batch_data is None:
            return samples
        
        try:
            # 嘗試從 loc_targets 中提取邊界框資訊
            if len(batch_data) >= 3:
                batch_images, batch_class_images, batch_loc_targets = batch_data[:3]
                
                if hasattr(batch_loc_targets, 'shape'):
                    batch_size = batch_loc_targets.shape[0]
                    
                    for i in range(batch_size):
                        # 從 loc_targets 中提取第一個有效的邊界框
                        loc_target = batch_loc_targets[i]  # [num_anchors, 4]
                        
                        if loc_target.numel() > 0:
                            # 找到第一個非零的邊界框
                            non_zero_mask = (loc_target != 0).any(dim=1)
                            if non_zero_mask.any():
                                first_valid_idx = non_zero_mask.nonzero()[0].item()
                                bbox = loc_target[first_valid_idx]  # [4]
                                
                                # 轉換為座標格式
                                x1, y1, x2, y2 = bbox.tolist()
                                point1 = (float(x1), float(y1))
                                point2 = (float(x2), float(y2))
                            else:
                                point1 = (0.0, 0.0)
                                point2 = (100.0, 100.0)
                        else:
                            point1 = (0.0, 0.0)
                            point2 = (100.0, 100.0)
                        
                        samples.append((i, 0, point1, point2))
                        
        except Exception as e:
            print(f"[WARNING] 替代提取方法失敗: {e}")
            # 最後的降級方案
            batch_size = 1
            samples.append((0, 0, (0.0, 0.0), (100.0, 100.0)))
        
        return samples
