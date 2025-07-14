from os2d.engine.objective import Os2dObjective
import torch
import torch.nn as nn

class LCPFinetuneCriterion(nn.Module):
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
                patch_mining_mode=False, batch_data=None):
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
        
        # 計算並加入輔助損失
        auxiliary_loss = self.compute_auxiliary_loss_from_batch(batch_data)
        total_loss = original_losses["loss"] + self.auxiliary_weight * auxiliary_loss
        
        # 更新損失字典
        lcp_losses = original_losses.copy()
        lcp_losses["loss"] = total_loss
        lcp_losses["auxiliary_loss"] = auxiliary_loss
        lcp_losses["original_loss"] = original_losses["loss"]
        
        if patch_mining_mode:
            return lcp_losses, losses_per_anchor
        else:
            return lcp_losses
    
    def compute_auxiliary_loss_from_batch(self, batch_data):
        """從批次數據計算輔助損失"""
        if batch_data is None or self.aux_net is None:
            return torch.tensor(0.0, requires_grad=True)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        total_aux_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        try:
            # 從 batch_data 中提取樣本資訊並計算輔助損失
            sample_count = 0
            
            for sample_info in self.extract_samples_from_batch(batch_data):
                image_id, class_id, point1, point2 = sample_info
                
                # 使用你現有的 aux_loss 方法
                sample_aux_loss = self.aux_net.aux_loss(
                    image_id, class_id, point1, point2
                )
                
                # 確保在正確設備上
                if sample_aux_loss.device != device:
                    sample_aux_loss = sample_aux_loss.to(device)
                
                total_aux_loss = total_aux_loss + sample_aux_loss
                sample_count += 1
            
            # 平均化
            if sample_count > 0:
                total_aux_loss = total_aux_loss / sample_count
                
        except Exception as e:
            print(f"[WARNING] 輔助損失計算失敗: {e}")
            total_aux_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return total_aux_loss
    
    def extract_samples_from_batch(self, batch_data):
        """
        從批次數據中提取樣本資訊
        根據 _prepare_batch 的返回格式進行實作
        """
        samples = []
        
        if batch_data is None or len(batch_data) < 9:
            return samples
        
        # 解包 batch_data
        # batch_data 格式: (batch_images, batch_class_images, batch_loc_targets, 
        #                  batch_class_targets, class_ids, class_image_sizes,
        #                  batch_box_inverse_transform, batch_boxes, batch_img_size)
        try:
            (batch_images, batch_class_images, batch_loc_targets, 
             batch_class_targets, class_ids, class_image_sizes,
             batch_box_inverse_transform, batch_boxes, batch_img_size) = batch_data
            
            batch_size = len(batch_boxes)
            
            for i in range(batch_size):
                # 從 batch_boxes 中提取邊界框資訊
                boxes = batch_boxes[i]
                
                if hasattr(boxes, 'bbox') and len(boxes.bbox) > 0:
                    # 取第一個有效的邊界框
                    bbox = boxes.bbox[0]  # [x1, y1, x2, y2]
                    
                    # 提取座標
                    x1, y1, x2, y2 = bbox.tolist() if hasattr(bbox, 'tolist') else bbox
                    point1 = (float(x1), float(y1))
                    point2 = (float(x2), float(y2))
                    
                    # 提取類別ID
                    if hasattr(boxes, 'get_field') and boxes.has_field('labels'):
                        labels = boxes.get_field('labels')
                        if len(labels) > 0:
                            class_id = int(labels[0])
                        else:
                            class_id = 0
                    else:
                        class_id = 0
                    
                    # 圖像ID使用批次索引
                    image_id = i
                    
                    samples.append((image_id, class_id, point1, point2))
                
                # 如果沒有有效邊界框，使用預設值
                else:
                    image_id = i
                    class_id = 0
                    point1 = (0.0, 0.0)
                    point2 = (100.0, 100.0)
                    samples.append((image_id, class_id, point1, point2))
                    
        except Exception as e:
            print(f"[WARNING] 批次數據解析失敗: {e}")
            # 降級到簡單實作
            batch_size = len(batch_data[0]) if len(batch_data) > 0 else 1
            for i in range(batch_size):
                samples.append((i, 0, (0.0, 0.0), (100.0, 100.0)))
        
        return samples
    
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
