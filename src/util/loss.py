from os2d.engine.objective import Os2dObjective
import torch
import torch.nn as nn
import numpy as np
import gc
from contextlib import contextmanager

@contextmanager
def memory_cleanup():
    """記憶體清理上下文管理器"""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

import csv
import os
from datetime import datetime

class LossLogger:
    def __init__(self, csv_path="loss_log.csv"):
        self.csv_path = csv_path
        # 如果檔案不存在，寫入標題
        if not os.path.exists(csv_path):
            with open(csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    "timestamp", "batch_idx", "original_loss", "auxiliary_loss", 
                    "normalized_auxiliary_loss", "total_loss", "normalization_factor",
                    "auxiliary_weight", "ema_original", "ema_auxiliary"
                ])
    
    def log_loss(self, batch_idx, original_loss, auxiliary_loss, normalized_auxiliary_loss, 
                 total_loss, normalization_factor, auxiliary_weight, ema_original, ema_auxiliary):
        """
        記錄損失數據到 CSV
        只有當所有損失都 > 0 時才記錄
        """
        # 檢查是否有任何損失為 0 或無效
        if (original_loss <= 0 or auxiliary_loss <= 0 or 
            normalized_auxiliary_loss <= 0 or total_loss <= 0):
            return False
        
        try:
            with open(self.csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    datetime.now().isoformat(),
                    batch_idx,
                    float(original_loss),
                    float(auxiliary_loss),
                    float(normalized_auxiliary_loss),
                    float(total_loss),
                    float(normalization_factor),
                    float(auxiliary_weight),
                    float(ema_original),
                    float(ema_auxiliary)
                ])
            return True
        except Exception as e:
            print(f"[LCP Logger] Error writing to CSV: {e}")
            return False
        
class LCPFinetuneCriterion(nn.Module):
    def __init__(self, original_criterion, aux_net, auxiliary_weight=1.0):
        """
        使用組合模式包裝 Os2dObjective
        
        Args:
            original_criterion: Os2dObjective 實例
            aux_net: 輔助網路實例
            auxiliary_weight: 輔助損失權重
        """
        super().__init__()
        self.original_criterion = original_criterion
        self.aux_net = aux_net
        self.auxiliary_weight = auxiliary_weight
        self.loss_logger = LossLogger()
    
    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets,
                cls_targets_remapped=None, cls_preds_for_neg=None,
                patch_mining_mode=False, batch_idx=None):
        """委託給原始 criterion 並加入輔助損失 - 損失正規化版本"""
        
        # 初始化指數移動平均值 (用於動態正規化)
        if not hasattr(self, 'loss_ema'):
            self.loss_ema = {
                'original': 1.0,
                'auxiliary': 1.0,
                'decay': 0.99
            }
        
        # 初始化統計計數器
        if not hasattr(self, 'loss_stats'):
            self.loss_stats = {
                'total_batches': 0,
                'valid_combinations': 0,
                'invalid_original': 0,
                'invalid_auxiliary': 0
            }
        
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
        self.get_dataloader().shuffle()
        auxiliary_loss = self.compute_auxiliary_loss_from_batch(batch_idx)
        original_loss = original_losses["loss"]
        
        # 更新統計計數器
        self.loss_stats['total_batches'] += 1
        
        # 🔴 有效性檢查：只有當兩個損失都有效時才計算
        VALIDITY_THRESHOLD = 1e-6
        original_is_valid = original_loss.item() > VALIDITY_THRESHOLD
        auxiliary_is_valid = auxiliary_loss.item() > VALIDITY_THRESHOLD
        
        print(f"[LCP Loss] Original loss: {original_loss:.6f} (valid: {original_is_valid})")
        print(f"[LCP Loss] Auxiliary loss: {auxiliary_loss:.6f} (valid: {auxiliary_is_valid})")
        
        # 只有當兩個損失都有效時才進行組合
        if original_is_valid and auxiliary_is_valid:
            print("[LCP Info] ✅ Both losses are valid, applying normalization")
            
            # 更新統計
            self.loss_stats['valid_combinations'] += 1
            
            # 🔴 更新指數移動平均值
            self.loss_ema['original'] = (self.loss_ema['decay'] * self.loss_ema['original'] + 
                                        (1 - self.loss_ema['decay']) * original_loss.item())
            self.loss_ema['auxiliary'] = (self.loss_ema['decay'] * self.loss_ema['auxiliary'] + 
                                        (1 - self.loss_ema['decay']) * auxiliary_loss.item())
            
            # 🔴 計算動態正規化係數
            # 目標：使輔助損失與原始損失在相同量級
            normalization_factor = self.loss_ema['original'] / (self.loss_ema['auxiliary'] + 1e-8)
            
            # 限制正規化係數在合理範圍內
            normalization_factor = max(0.05, min(2.0, normalization_factor))
            
            # 🔴 應用正規化
            normalized_auxiliary = auxiliary_loss * normalization_factor
            
            # 根據 LCP 論文公式計算總損失
            total_loss = original_loss + self.auxiliary_weight * normalized_auxiliary
            
            print(f"[LCP Normalization] EMA Original: {self.loss_ema['original']:.6f}")
            print(f"[LCP Normalization] EMA Auxiliary: {self.loss_ema['auxiliary']:.6f}")
            print(f"[LCP Normalization] Normalization factor: {normalization_factor:.6f}")
            print(f"[LCP Normalization] Normalized auxiliary: {normalized_auxiliary:.6f}")
            print(f"[LCP Loss] Auxiliary weight: {self.auxiliary_weight}")
            print(f"[LCP Loss] Total loss: {total_loss:.6f}")
            
            # 🔴 記錄有效的損失數據到 CSV
            log_success = self.loss_logger.log_loss(
                batch_idx=batch_idx if batch_idx is not None else self.loss_stats['total_batches'],
                original_loss=original_loss.item(),
                auxiliary_loss=auxiliary_loss.item(),
                normalized_auxiliary_loss=normalized_auxiliary.item(),
                total_loss=total_loss.item(),
                normalization_factor=normalization_factor,
                auxiliary_weight=self.auxiliary_weight,
                ema_original=self.loss_ema['original'],
                ema_auxiliary=self.loss_ema['auxiliary']
            )


            # 🔴 關鍵修改：只包含 Tensor 類型的損失值
            lcp_losses = original_losses.copy()
            lcp_losses["loss"] = total_loss
            lcp_losses["auxiliary_loss"] = auxiliary_loss
            lcp_losses["normalized_auxiliary_loss"] = normalized_auxiliary
            lcp_losses["original_loss"] = original_loss
            
            # 🔴 將正規化係數轉換為 Tensor 以避免類型錯誤
            lcp_losses["normalization_factor"] = torch.tensor(normalization_factor, 
                                                            device=original_loss.device, 
                                                            dtype=original_loss.dtype)
            
            # 🔴 移除布林值，改為存儲在類屬性中
            self.last_loss_valid = True
            
        else:
            # 任何一個損失無效時，跳過組合
            if not original_is_valid:
                print("[LCP Warning] ❌ Original loss is invalid (≤ 1e-6)")
                self.loss_stats['invalid_original'] += 1
            if not auxiliary_is_valid:
                print("[LCP Warning] ❌ Auxiliary loss is invalid (≤ 1e-6)")
                self.loss_stats['invalid_auxiliary'] += 1
            
            print("[LCP Info] 🚫 Loss combination skipped - invalid losses detected")
            
            # 返回原始損失，標記為無效
            lcp_losses = original_losses.copy()
            lcp_losses["auxiliary_loss"] = auxiliary_loss
            lcp_losses["original_loss"] = original_loss
            
            # 🔴 移除布林值，改為存儲在類屬性中
            self.last_loss_valid = False
            
            # 🔴 關鍵修改：使用原始損失而不是極小值，避免訓練停滯
            # 當輔助損失無效時，仍然可以用原始損失進行訓練
            lcp_losses["loss"] = original_loss
            
            # 添加零值的正規化係數作為 Tensor
            lcp_losses["normalization_factor"] = torch.tensor(0.0, 
                                                            device=original_loss.device, 
                                                            dtype=original_loss.dtype)
        
        # 🔴 定期輸出統計信息
        if self.loss_stats['total_batches'] % 100 == 0:
            self.print_loss_statistics()
        
        if patch_mining_mode:
            return lcp_losses, losses_per_anchor
        else:
            return lcp_losses

    def print_loss_statistics(self):
        """列印損失統計信息"""
        stats = self.loss_stats
        total = stats['total_batches']
        
        if total > 0:
            print(f"\n[LCP Statistics] After {total} batches:")
            print(f"  Valid combinations: {stats['valid_combinations']} ({stats['valid_combinations']/total*100:.1f}%)")
            print(f"  Invalid original: {stats['invalid_original']} ({stats['invalid_original']/total*100:.1f}%)")
            print(f"  Invalid auxiliary: {stats['invalid_auxiliary']} ({stats['invalid_auxiliary']/total*100:.1f}%)")
            print(f"  Overall validity rate: {stats['valid_combinations']/total*100:.1f}%")
            
            # 如果有效率過低，提供建議
            if stats['valid_combinations']/total < 0.3:
                print(f"  ⚠️  WARNING: Low validity rate, consider checking data pipeline")
            print()

    def get_loss_status(self):
        """獲取當前損失狀態"""
        return {
            'is_valid': getattr(self, 'last_loss_valid', False),
            'ema_original': self.loss_ema['original'] if hasattr(self, 'loss_ema') else 0.0,
            'ema_auxiliary': self.loss_ema['auxiliary'] if hasattr(self, 'loss_ema') else 0.0,
            'statistics': self.loss_stats if hasattr(self, 'loss_stats') else {}
        }

    def get_image_ids_for_batch_index(self, batch_idx):
        """獲取指定批次索引的圖像 ID"""
        return self.aux_net.get_contextual_roi_align().dataloader.get_image_ids_for_batch_index(batch_idx)

    def get_dataloader(self):
        """獲取數據加載器"""
        return self.aux_net.get_contextual_roi_align().dataloader

    def get_database(self):
        return self.aux_net.get_db()

    def compute_auxiliary_loss_from_batch(self, batch_idx):
      """從批次數據計算輔助損失 - 極致 GPU RAM 優化版本"""
      
      # 記憶體監控函數
      def log_memory(stage):
          if torch.cuda.is_available():
              allocated = torch.cuda.memory_allocated(0) / 1024**3
              reserved = torch.cuda.memory_reserved(0) / 1024**3
              print(f"[Memory] {stage}: 已分配 {allocated:.2f} GB, 已保留 {reserved:.2f} GB")
      
      # 強制記憶體清理
      def force_cleanup():
          if torch.cuda.is_available():
              torch.cuda.empty_cache()
              torch.cuda.synchronize()
          gc.collect()
      
      print(f"[LCP Debug] Computing auxiliary loss...")
      print(f"[LCP Debug] batch_idx is None: {batch_idx is None}")
      print(f"[LCP Debug] aux_net is None: {self.aux_net is None}")
      
      log_memory("函數開始")

      if batch_idx is None or self.aux_net is None:
          print(f"[LCP Debug] Returning zero auxiliary loss")
          return torch.tensor(0.0, requires_grad=True)
      
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      
      try:
          # 🔴 階段1: 獲取基本資料並立即清理
          with memory_cleanup():
              image_ids = self.get_image_ids_for_batch_index(batch_idx)
              log_memory("獲取 image_ids 後")
          
          # 🔴 階段2: 分批次處理資料
          batch = self.get_dataloader().get_batch(batch_idx)
          
          # 立即解包並清理原始 batch
          images, class_images, loc_targets, cls_targets, class_ids, class_sizes, transforms, boxes, img_sizes = batch
          del batch
          force_cleanup()
          log_memory("批次資料解包後")

          # 調試資訊
          print(f"[LCP Debug] class_ids type: {type(class_ids)}")
          print(f"[LCP Debug] class_ids length: {len(class_ids) if hasattr(class_ids, '__len__') else 'No length'}")
          if len(class_ids) > 0:
              print(f"[LCP Debug] class_ids[0] type: {type(class_ids[0])}")

          # 🔴 階段3: 極致樣本數量限制
          db = self.get_database()
          
          # 更嚴格的限制：最多 8 個樣本
          MAX_SAMPLES = min( 8 , len(image_ids) )
          samples_collected = 0
          total_aux_loss = torch.tensor(0.0, device=device, requires_grad=True)
          
          # 🔴 直接處理樣本，不先收集
          for i, image_id in enumerate(image_ids):
              if samples_collected >= MAX_SAMPLES:
                  break
              
              # 每處理5個 image_id 就清理一次記憶體
              if i % 5 == 0:
                  force_cleanup()
                  log_memory(f"處理 image_id {i}")
              
              # 處理 class_ids 的不同結構
              if isinstance(class_ids[i], (list, tuple, np.ndarray)):
                  current_class_ids = class_ids[i]
              else:
                  current_class_ids = [class_ids[i]]
              
              for j, class_id in enumerate(current_class_ids):
                  if samples_collected >= MAX_SAMPLES:
                      break
                      
                  class_id = int(class_id)
                  
                  try:
                      # 🔴 逐個獲取並立即使用資料
                      point1_xs = db.get_specific_data(image_id, class_id, 'point1_x')
                      point1_ys = db.get_specific_data(image_id, class_id, 'point1_y')
                      point2_xs = db.get_specific_data(image_id, class_id, 'point2_x')
                      point2_ys = db.get_specific_data(image_id, class_id, 'point2_y')

                      # 檢查數據完整性
                      if not (point1_xs and point1_ys and point2_xs and point2_ys):
                          continue
                      
                      # 只處理第一個有效樣本
                      min_length = min(len(point1_xs), len(point1_ys), len(point2_xs), len(point2_ys))
                      
                      if min_length > 0:
                          try:
                              # 只取第一個樣本
                              point1 = (float(point1_xs[0]), float(point1_ys[0]))
                              point2 = (float(point2_xs[0]), float(point2_ys[0]))
                              
                              # 🔴 立即計算輔助損失
                              with memory_cleanup():
                                  sample_aux_loss = self.aux_net.aux_loss(
                                      image_id, 
                                      class_id, 
                                      point1, 
                                      point2
                                  )
                                  
                                  # 確保在正確設備上
                                  if sample_aux_loss.device != device:
                                      sample_aux_loss = sample_aux_loss.to(device)
                                  
                                  # 立即累積並分離
                                  total_aux_loss = total_aux_loss + sample_aux_loss.detach()
                                  samples_collected += 1
                                  
                                  print(f"[LCP Debug] Sample {samples_collected}: aux_loss = {sample_aux_loss.item():.6f}")
                                  
                                  # 立即刪除樣本損失
                                  del sample_aux_loss
                              
                          except (ValueError, TypeError) as e:
                              print(f"[LCP Debug] Error processing sample: {e}")
                              continue
                      
                      # 🔴 立即清理資料庫查詢結果
                      del point1_xs, point1_ys, point2_xs, point2_ys
                      
                      if min_length > 0:
                          break
                          
                  except Exception as e:
                      print(f"[LCP Debug] Error accessing data for image_id={image_id}, class_id={class_id}: {e}")
                      continue
          
          # 🔴 階段4: 清理所有批次相關資料
          del images, class_images, loc_targets, cls_targets, class_ids, class_sizes, transforms, boxes, img_sizes
          force_cleanup()
          log_memory("批次資料清理後")
          
          # 🔴 階段5: 計算最終結果
          if samples_collected > 0:
              average_aux_loss = total_aux_loss / samples_collected
              print(f"[LCP Debug] Total samples processed: {samples_collected}/{MAX_SAMPLES}")
              print(f"[LCP Debug] Average auxiliary loss: {average_aux_loss.item():.6f}")
              
              # 🔴 確保返回的張量不保留計算圖
              result = average_aux_loss.clone().detach().requires_grad_(True)
              del average_aux_loss, total_aux_loss
              
              force_cleanup()
              log_memory("函數結束")
              
              return result
          else:
              print(f"[LCP Debug] No valid samples processed, returning zero loss")
              del total_aux_loss
              force_cleanup()
              return torch.tensor(0.0, device=device, requires_grad=True)
              
      except Exception as e:
          print(f"[LCP Error] Failed to compute auxiliary loss: {e}")
          import traceback
          traceback.print_exc()
          
          # 清理所有可能的變數
          try:
              del total_aux_loss
          except:
              pass
          
          force_cleanup()
          return torch.tensor(0.0, device=device, requires_grad=True)
    
      finally:
          # 🔴 最終清理：確保所有資源都被釋放
          force_cleanup()
          log_memory("最終清理後")

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