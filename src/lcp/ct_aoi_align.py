import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from os2d.utils import get_image_size_after_resize_preserving_aspect_ratio
from torch.utils.checkpoint import checkpoint
import torch
import gc

class ContextAoiAlign:
    def __init__(self, db, dataloader=None, transform_image=None, net=None, cfg=None):
        self.db = db
        self.dataloader = dataloader
        self.transform_image = transform_image
        self.net = net
        self.cfg = cfg
        self.backprop = False
        self.use_checkpoint = True  # 預設啟用檢查點技術
        self.checkpoint_strategy = 'adaptive'

    def enable_gradient_checkpoint(self, strategy='adaptive'):
        """啟用梯度追蹤和檢查點優化"""
        self.backprop = True
        self.use_checkpoint = True
        self.checkpoint_strategy = strategy
        print(f"[ContextAoiAlign] 梯度檢查點已啟用，策略: {strategy}")
    
    def disable_gradient_checkpoint(self):
        """關閉梯度追蹤和檢查點"""
        self.backprop = False
        self.use_checkpoint = False
        print(f"[ContextAoiAlign] 梯度檢查點已關閉")
    
    def set_checkpoint_strategy(self, strategy):
        """設置檢查點策略"""
        valid_strategies = ['standard', 'segmented', 'adaptive']
        if strategy in valid_strategies:
            self.checkpoint_strategy = strategy
            print(f"[ContextAoiAlign] 檢查點策略設為: {strategy}")
        else:
            print(f"[ERROR] 無效的策略: {strategy}，有效選項: {valid_strategies}")

    def bilinear_interpolate(self, feature_map, x, y):
        """
        雙線性插值實現
        
        Args:
            feature_map: 特徵圖 [C, H, W]
            x: x座標 (浮點數，可為 tensor)
            y: y座標 (浮點數，可為 tensor)
        
        Returns:
            插值後的特徵值 [C]
        """
        C, H, W = feature_map.shape
        
        # 確保輸入為 tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=feature_map.device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32, device=feature_map.device)
        
        # 邊界處理：確保座標在有效範圍內
        x = torch.clamp(x, 0.0, W - 1.0)
        y = torch.clamp(y, 0.0, H - 1.0)
        
        # 找到4個鄰近的網格點座標
        x0 = torch.floor(x).long()  # 左邊界
        x1 = torch.clamp(x0 + 1, 0, W - 1).long()  # 右邊界
        y0 = torch.floor(y).long()  # 上邊界
        y1 = torch.clamp(y0 + 1, 0, H - 1).long()  # 下邊界
        
        # 計算相對位置（0-1之間的小數部分）
        dx = x - x0.float()  # x方向的相對位置
        dy = y - y0.float()  # y方向的相對位置
        
        # 計算4個角點的權重
        w00 = (1.0 - dx) * (1.0 - dy)  # 左上角權重
        w01 = (1.0 - dx) * dy          # 左下角權重
        w10 = dx * (1.0 - dy)          # 右上角權重
        w11 = dx * dy                  # 右下角權重
        
        # 提取4個角點的特徵值
        f00 = feature_map[:, y0, x0]  # 左上角特徵
        f01 = feature_map[:, y1, x0]  # 左下角特徵
        f10 = feature_map[:, y0, x1]  # 右上角特徵
        f11 = feature_map[:, y1, x1]  # 右下角特徵
        
        # 雙線性插值計算
        interpolated = w00 * f00 + w01 * f01 + w10 * f10 + w11 * f11
        
        return interpolated
    def calculate_spatial_scale(self, image_id, feature_map):
        # 獲取原圖尺寸
        original_image = self.dataloader._get_dataset_image_by_id(image_id)
        orig_h, orig_w = original_image.size[1], original_image.size[0]
        
        # 獲取特徵圖尺寸
        feat_h, feat_w = feature_map.shape[1], feature_map.shape[2]
        
        # 計算縮放比例
        spatial_scale = feat_w / orig_w  # 或 feat_h / orig_h
        
        return spatial_scale
    
    def roi_align_single_roi(self, feature_map, roi_box, output_size=(7, 7), sampling_ratio=2, spatial_scale=1.0):
        """
        對單個 ROI 執行 RoI Align
        
        Args:
            feature_map: 特徵圖 [C, H, W]
            roi_box: ROI 邊界框 ((x1, y1), (x2, y2))
            output_size: 輸出特徵圖大小 (pooled_h, pooled_w)
            sampling_ratio: 每個 bin 每個方向的採樣點數
            spatial_scale: 特徵圖相對於原圖的縮放比例
        
        Returns:
            提取的特徵 [C, pooled_h, pooled_w]
        """
        C, H, W = feature_map.shape
        (x1, y1), (x2, y2) = roi_box
        
        # 確保座標為浮點數並應用空間縮放
        x1, y1, x2, y2 = float(x1) * spatial_scale, float(y1) * spatial_scale, \
                        float(x2) * spatial_scale, float(y2) * spatial_scale
        
        # 計算 RoI 的寬度和高度
        roi_width = x2 - x1
        roi_height = y2 - y1
        
        # 計算每個 bin 的大小
        pooled_h, pooled_w = output_size
        bin_size_h = roi_height / pooled_h
        bin_size_w = roi_width / pooled_w
        
        # 初始化輸出特徵張量
        pooled_features = torch.zeros(C, pooled_h, pooled_w, 
                                    dtype=feature_map.dtype, 
                                    device=feature_map.device)
        
        # 計算每個 bin 的採樣點數量
        roi_bin_grid_h = sampling_ratio if sampling_ratio > 0 else math.ceil(roi_height / pooled_h)
        roi_bin_grid_w = sampling_ratio if sampling_ratio > 0 else math.ceil(roi_width / pooled_w)
        count = roi_bin_grid_h * roi_bin_grid_w
        
        # 遍歷每個輸出 bin
        for ph in range(pooled_h):
            for pw in range(pooled_w):
                # 計算當前 bin 的邊界
                hstart = y1 + ph * bin_size_h
                wstart = x1 + pw * bin_size_w
                
                # 收集該 bin 內所有採樣點的特徵值
                bin_values = []
                
                # 在 bin 內進行網格採樣
                for iy in range(roi_bin_grid_h):
                    for ix in range(roi_bin_grid_w):
                        # 計算採樣點座標（在 bin 內均勻分布）
                        sample_y = hstart + (iy + 0.5) * bin_size_h / roi_bin_grid_h
                        sample_x = wstart + (ix + 0.5) * bin_size_w / roi_bin_grid_w
                        
                        # 使用雙線性插值獲取該點的特徵值
                        interpolated_value = self.bilinear_interpolate(feature_map, sample_x, sample_y)
                        bin_values.append(interpolated_value)
                
                # 對該 bin 內所有採樣點進行平均池化
                if bin_values:
                    pooled_features[:, ph, pw] = torch.stack(bin_values).mean(dim=0)
                else:
                    # 如果沒有有效採樣點，設為0
                    pooled_features[:, ph, pw] = 0
        
        return pooled_features

    def get_feature_map(self, image_id):
        if self.dataloader is None or self.net is None:
            raise ValueError("Dataloader and network must be provided")

        # 1. 獲取原始圖像
        input_image = self.dataloader._get_dataset_image_by_id(image_id)

        # 2. 計算保持長寬比的 resize 尺寸
        h, w = get_image_size_after_resize_preserving_aspect_ratio(
            h=input_image.size[1],
            w=input_image.size[0],
            target_size=1500
        )
        
        # 3. resize 圖像
        input_image = input_image.resize((w, h))

        # 4. 圖像轉換為 tensor 並標準化
        input_image_th = self.transform_image(input_image)
        input_image_th = input_image_th.unsqueeze(0)

        # 5. 設備管理（建議使用 CPU 避免 GPU OOM）
        if self.cfg.is_cuda and torch.cuda.is_available():
            input_image_th = input_image_th.cuda()
            device = 'cuda'
        else:
            input_image_th = input_image_th.cpu()
            device = 'cpu'
    
        feature_map = self.net.net_feature_maps(input_image_th)
        feature_map = feature_map.squeeze(0)

        return feature_map
    
    def get_feature_map_no_grad(self, image_id):
        """無梯度版本的特徵圖提取"""
        if self.dataloader is None or self.net is None:
            raise ValueError("Dataloader and network must be provided")
        
        # 獲取原始圖像並進行預處理
        input_image = self.dataloader._get_dataset_image_by_id(image_id)
        h, w = get_image_size_after_resize_preserving_aspect_ratio(
            h=input_image.size[1], w=input_image.size[0], target_size=1500
        )
        input_image = input_image.resize((w, h))
        input_image_th = self.transform_image(input_image).unsqueeze(0)
        
        # 設備管理
        if self.cfg.is_cuda and torch.cuda.is_available():
            input_image_th = input_image_th.cuda()
        
        # 無梯度特徵提取
        with torch.no_grad():
            feature_map = self.net.net_feature_maps(input_image_th)
        
        return feature_map.squeeze(0)
    
    def extract_roi_features_by_ids_in_no_grad(self, image_id, class_id, output_size=(7, 7)):
        t_x1s = self.db.get_specific_data(image_id, class_id, "point1_x")
        t_y1s = self.db.get_specific_data(image_id, class_id, "point1_y")
        t_x2s = self.db.get_specific_data(image_id, class_id, "point2_x")
        t_y2s = self.db.get_specific_data(image_id, class_id, "point2_y")
        
        c_x1s = self.db.get_specific_data(image_id, class_id, "context_roi_point1_x")
        c_y1s = self.db.get_specific_data(image_id, class_id, "context_roi_point1_y")
        c_x2s = self.db.get_specific_data(image_id, class_id, "context_roi_point2_x")
        c_y2s = self.db.get_specific_data(image_id, class_id, "context_roi_point2_y")
        
        features_list = []
        
        for tx1, ty1, tx2, ty2, cx1, cy1, cx2, cy2 in zip(
            t_x1s, t_y1s, t_x2s, t_y2s, c_x1s, c_y1s, c_x2s, c_y2s
        ):
            point1 = (float(tx1), float(ty1))
            point2 = (float(tx2), float(ty2))
            context_roi_point1 = (float(cx1), float(cy1))
            context_roi_point2 = (float(cx2), float(cy2))
            
            # 執行 ROI Align
            features = self.extract_roi_align_no_grad(
                image_id, class_id, point1, point2,
                context_roi_point1, context_roi_point2, output_size
            )
            
            features_list.append({
                'image_id': image_id,
                'class_id': class_id,
                'truth_box': (point1, point2),
                'context_roi': (context_roi_point1, context_roi_point2),
                'features': features
            })
        
        return features_list
    def extract_roi_align_no_grad(self, image_id, class_id, point1, point2, 
                             context_roi_point1, context_roi_point2, output_size=(7, 7)):
        """無梯度版本的 ROI 特徵提取"""
        # 獲取特徵圖
        feature_map = self.get_feature_map_no_grad(image_id)
        
        # 計算空間縮放比例
        spatial_scale = self.calculate_spatial_scale(image_id, feature_map)
        
        if context_roi_point1 is None or context_roi_point2 is None:
            return None
        
        # 執行 RoI Align
        truth_box = (point1, point2)
        context_roi_box = (context_roi_point1, context_roi_point2)
        
        truth_feature = self.roi_align_single_roi(
            feature_map=feature_map,
            roi_box=truth_box,
            output_size=output_size,
            sampling_ratio=2,
            spatial_scale=spatial_scale
        )
        
        context_feature = self.roi_align_single_roi(
            feature_map=feature_map,
            roi_box=context_roi_box,
            output_size=output_size,
            sampling_ratio=2,
            spatial_scale=spatial_scale
        )
        
        # 論文公式：F_O = RoIAlign(F, B) + RoIAlign(F, C)
        combined_feature = truth_feature + context_feature
        
        return {
            'truth_feature': truth_feature,
            'context_feature': context_feature,
            'combined_feature': combined_feature
        }
    
    def _forward_with_checkpoint(self, input_image_th, device):
        """使用梯度檢查點的前向傳播 - 修復版本"""
        checkpoint_strategy = getattr(self, 'checkpoint_strategy', 'adaptive')
        
        print(f"[CHECKPOINT] 使用 {checkpoint_strategy} 檢查點策略")
        
        # 確保網路在正確的設備上
        target_device = input_image_th.device
        self.net.net_feature_maps = self.net.net_feature_maps.to(target_device)
        
        try:
            if checkpoint_strategy == 'standard':
                return self._standard_checkpoint(input_image_th)
            elif checkpoint_strategy == 'segmented':
                return self._segmented_checkpoint(input_image_th, device)
            elif checkpoint_strategy == 'adaptive':
                return self._adaptive_checkpoint(input_image_th, device)
            else:
                # 預設使用標準檢查點
                return self._standard_checkpoint(input_image_th)
        except Exception as e:
            print(f"[CHECKPOINT] 檢查點失敗，使用直接前向傳播: {e}")
            # 最終回退：不使用檢查點
            return self.net.net_feature_maps(input_image_th)

    def _standard_checkpoint(self, input_image_th):
        """標準檢查點：整個 backbone 作為一個檢查點"""
        return checkpoint(self.net.net_feature_maps, input_image_th)

    def _segmented_checkpoint(self, input_image_th, device):
        """分段檢查點：確保所有段落都在正確的設備上"""
        segments = self._get_backbone_segments()
        
        # 關鍵修復：確保所有段落都移動到正確的設備
        target_device = input_image_th.device  # 使用輸入張量的設備
        
        x = input_image_th
        for i, segment in enumerate(segments):
            print(f"[CHECKPOINT] 處理段落 {i+1}/{len(segments)}")
            
            # 確保段落在正確的設備上
            segment = segment.to(target_device)
            
            # 使用檢查點
            x = checkpoint(segment, x, use_reentrant=False)
            
            # 清理記憶體
            gc.collect()
            if target_device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return x

    def _adaptive_checkpoint(self, input_image_th, device):
        """自適應檢查點：添加設備檢查和錯誤處理"""
        try:
            # 檢查輸入張量和網路的設備
            input_device = input_image_th.device
            
            # 確保整個網路在正確的設備上
            self.net.net_feature_maps = self.net.net_feature_maps.to(input_device)
            
            if device == 'cuda' and torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                if gpu_memory_used > 0.8:
                    print(f"[CHECKPOINT] GPU 記憶體使用率高 ({gpu_memory_used:.1%})，使用分段檢查點")
                    return self._segmented_checkpoint(input_image_th, device)
                else:
                    print(f"[CHECKPOINT] GPU 記憶體使用率正常 ({gpu_memory_used:.1%})，使用標準檢查點")
                    return self._standard_checkpoint(input_image_th)
            else:
                return self._standard_checkpoint(input_image_th)
                
        except Exception as e:
            print(f"[CHECKPOINT] 自適應檢查點失敗: {e}")
            # 回退策略：確保設備一致後使用標準檢查點
            try:
                self.net.net_feature_maps = self.net.net_feature_maps.to(input_image_th.device)
                return self._standard_checkpoint(input_image_th)
            except Exception as e2:
                print(f"[CHECKPOINT] 標準檢查點也失敗: {e2}")
                # 最後回退：不使用檢查點，直接前向傳播
                self.net.net_feature_maps = self.net.net_feature_maps.to(input_image_th.device)
                return self.net.net_feature_maps(input_image_th)
    
    def _get_backbone_segments(self):
        """改進的網路分段策略 - 確保設備一致性"""
        segments = []
        
        try:
            # 獲取原始網路的設備
            original_device = next(self.net.net_feature_maps.parameters()).device
            
            # 創建段落列表（不立即移動設備，在使用時再移動）
            layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
            
            for layer_name in layer_names:
                if hasattr(self.net.net_feature_maps, layer_name):
                    layer = getattr(self.net.net_feature_maps, layer_name)
                    segments.append(layer)
            
            print(f"[CHECKPOINT] 網路分成 {len(segments)} 個段落")
            return segments
            
        except Exception as e:
            print(f"[CHECKPOINT] 網路分段失敗，使用整個網路: {e}")
            return [self.net.net_feature_maps]

    def _create_initial_conv_block(self):
        """創建初始卷積塊"""
        import torch.nn as nn
        
        class InitialConvBlock(nn.Module):
            def __init__(self, net_feature_maps):
                super().__init__()
                self.conv1 = net_feature_maps.conv1
                if hasattr(net_feature_maps, 'bn1'):
                    self.bn1 = net_feature_maps.bn1
                if hasattr(net_feature_maps, 'relu'):
                    self.relu = net_feature_maps.relu
                if hasattr(net_feature_maps, 'maxpool'):
                    self.maxpool = net_feature_maps.maxpool
            
            def forward(self, x):
                x = self.conv1(x)
                if hasattr(self, 'bn1'):
                    x = self.bn1(x)
                if hasattr(self, 'relu'):
                    x = self.relu(x)
                if hasattr(self, 'maxpool'):
                    x = self.maxpool(x)
                return x
        
        return InitialConvBlock(self.net.net_feature_maps)

    def roi_align(self, image_id, class_id, point1, point2, context_roi_point1, context_roi_point2, output_size=(7, 7)):
        # 1. 獲取特徵圖
        feature_map = self.get_feature_map(image_id)  # [C, H, W]
        
        # 2. 計算 spatial_scale
        spatial_scale = self.calculate_spatial_scale(image_id, feature_map)
        
        # 3. 準備 ROI boxes（你的座標已經是原圖座標，直接使用）
        truth_box = (point1, point2)
        context_roi_box = (context_roi_point1, context_roi_point2)
        
        # 4. 執行 RoI Align
        truth_feature = self.roi_align_single_roi(
            feature_map=feature_map,
            roi_box=truth_box,
            output_size=(7, 7),        # 固定維度
            sampling_ratio=2,          # 每個 bin 2×2 採樣點
            spatial_scale=spatial_scale # 自動計算的縮放比例
        )
        
        context_feature = self.roi_align_single_roi(
            feature_map=feature_map,
            roi_box=context_roi_box,
            output_size=(7, 7),
            sampling_ratio=2,
            spatial_scale=spatial_scale
        )
        
        # 5. 論文公式：F_O = RoIAlign(F, B) + RoIAlign(F, C)
        combined_feature = truth_feature + context_feature
        
        # print(f"[LOG] Extracted features for image_id {image_id}, class_id {class_id}\
        #         \nTruth Feature Shape: {truth_feature.shape}, Context Feature Shape: {context_feature.shape}, Combined Feature Shape: {combined_feature.shape}\
        #         \nTruth Box: {truth_box}, Context ROI Box: {context_roi_box}\
        #         \nSpatial Scale: {spatial_scale}\
        #         \nOutput Size: {output_size}\
        #       ")

        return {
            'truth_feature': truth_feature,
            'context_feature': context_feature, 
            'combined_feature': combined_feature
        }


    def extract_roi_features_for_all(self, output_size=(7, 7)):
        """
        為所有圖像和類別提取 ROI 特徵
        
        Args:
            output_size: 輸出特徵圖大小
        
        Returns:
            list: 所有特徵的列表
        """
        image_ids = list(map(int, self.db.get_image_ids()))
        sorted_image_ids = sorted(image_ids)
        all_features = []
        
        for image_id in sorted_image_ids:
            get_class_id = self.db.get_class_ids_by_image_id(image_id)
            class_ids = list(map(int, get_class_id.keys()))
            
            for class_id in class_ids:
                features = self.extract_roi_features_by_ids(image_id, class_id, output_size)
                all_features.extend(features)
        
        return all_features

    def if_same_point( self, point1, point2 ):
        x1, y1 = point1
        x2, y2 = point2
        return abs(x1 - x2) < 1e-5 and abs(y1 - y2) < 1e-5
    
    def extract_roi_features_specific(self, image_id, class_id, point1, point2, output_size=(7, 7)):
        """
        為特定圖像和類別提取 ROI 特徵
        
        Args:
            image_id: 圖像ID
            class_id: 類別ID
            [point1, point2]: 真實邊界框的兩個點 (轉換後)
            output_size: 輸出特徵圖大小
        
        Returns:
            list: 該圖像類別的所有特徵
        """
        # 獲取所有相關數據
        t_x1s = self.db.get_specific_data(image_id, class_id, "point1_x")
        t_y1s = self.db.get_specific_data(image_id, class_id, "point1_y")
        t_x2s = self.db.get_specific_data(image_id, class_id, "point2_x")
        t_y2s = self.db.get_specific_data(image_id, class_id, "point2_y")
        
        c_x1s = self.db.get_specific_data(image_id, class_id, "context_roi_point1_x")
        c_y1s = self.db.get_specific_data(image_id, class_id, "context_roi_point1_y")
        c_x2s = self.db.get_specific_data(image_id, class_id, "context_roi_point2_x")
        c_y2s = self.db.get_specific_data(image_id, class_id, "context_roi_point2_y")
        
        features_list = []
        
        for tx1, ty1, tx2, ty2, cx1, cy1, cx2, cy2 in zip(
            t_x1s, t_y1s, t_x2s, t_y2s, c_x1s, c_y1s, c_x2s, c_y2s
        ):
            _point1 = (float(tx1), float(ty1))
            _point2 = (float(tx2), float(ty2))

            if not self.if_same_point(point1, _point1) or not self.if_same_point(point2, _point2):
                continue
            
            context_roi_point1 = (float(cx1), float(cy1))
            context_roi_point2 = (float(cx2), float(cy2))
            
            # 執行 ROI Align
            features = self.roi_align(
                image_id, class_id, _point1, _point2,
                context_roi_point1, context_roi_point2, output_size
            )
            
            features_list.append({
                'image_id': image_id,
                'class_id': class_id,
                'truth_box': (point1, point2),
                'context_roi': (context_roi_point1, context_roi_point2),
                'features': features
            })
        
        return features_list

    def extract_roi_features_by_ids(self, image_id, class_id, output_size=(7, 7)):
        """
        為特定圖像和類別提取 ROI 特徵
        
        Args:
            image_id: 圖像ID
            class_id: 類別ID
            output_size: 輸出特徵圖大小
        
        Returns:
            list: 該圖像類別的所有特徵
        """
        # 獲取所有相關數據
        t_x1s = self.db.get_specific_data(image_id, class_id, "point1_x")
        t_y1s = self.db.get_specific_data(image_id, class_id, "point1_y")
        t_x2s = self.db.get_specific_data(image_id, class_id, "point2_x")
        t_y2s = self.db.get_specific_data(image_id, class_id, "point2_y")
        
        c_x1s = self.db.get_specific_data(image_id, class_id, "context_roi_point1_x")
        c_y1s = self.db.get_specific_data(image_id, class_id, "context_roi_point1_y")
        c_x2s = self.db.get_specific_data(image_id, class_id, "context_roi_point2_x")
        c_y2s = self.db.get_specific_data(image_id, class_id, "context_roi_point2_y")
        
        features_list = []
        
        for tx1, ty1, tx2, ty2, cx1, cy1, cx2, cy2 in zip(
            t_x1s, t_y1s, t_x2s, t_y2s, c_x1s, c_y1s, c_x2s, c_y2s
        ):
            point1 = (float(tx1), float(ty1))
            point2 = (float(tx2), float(ty2))
            context_roi_point1 = (float(cx1), float(cy1))
            context_roi_point2 = (float(cx2), float(cy2))
            
            # 執行 ROI Align
            features = self.roi_align(
                image_id, class_id, point1, point2,
                context_roi_point1, context_roi_point2, output_size
            )
            
            features_list.append({
                'image_id': image_id,
                'class_id': class_id,
                'truth_box': (point1, point2),
                'context_roi': (context_roi_point1, context_roi_point2),
                'features': features
            })
        
        return features_list

    def compute_roi_region_by_ids(self, image_id, class_id):
        """原有方法保持不變"""
        t_x1s = self.db.get_specific_data(image_id, class_id, "point1_x")
        t_y1s = self.db.get_specific_data(image_id, class_id, "point1_y")
        t_x2s = self.db.get_specific_data(image_id, class_id, "point2_x")
        t_y2s = self.db.get_specific_data(image_id, class_id, "point2_y")

        d_x1s = self.db.get_specific_data(image_id, class_id, "detection_point1_x")
        d_y1s = self.db.get_specific_data(image_id, class_id, "detection_point1_y")
        d_x2s = self.db.get_specific_data(image_id, class_id, "detection_point2_x")
        d_y2s = self.db.get_specific_data(image_id, class_id, "detection_point2_y")

        for tx1, ty1, tx2, ty2, dx1, dy1, dx2, dy2 in zip(t_x1s, t_y1s, t_x2s, t_y2s, d_x1s, d_y1s, d_x2s, d_y2s):
            point1 = (float(tx1), float(ty1))
            point2 = (float(tx2), float(ty2))
            
            # 計算 IoU
            truth_box = (point1, point2)
            detect_box = ((float(dx1), float(dy1)), (float(dx2), float(dy2)))
            iou = self.db.compute_iou_for_pair(truth_box, detect_box)
            
            # 計算 context roi
            context_roi_point1 = (min(float(tx1), float(dx1)), min(float(ty1), float(dy1)))
            context_roi_point2 = (max(float(tx2), float(dx2)), max(float(ty2), float(dy2)))
            
            # 如果 IoU 太低，設置無效的 context roi
            if iou <= 0.5:
                context_roi_point1 = (-1, -1)
                context_roi_point2 = (-1, -1)
            
            self.db.write_context_roi_to_db(
                image_id, class_id, point1, point2,
                context_roi_point1, context_roi_point2
            )

    # 其他原有方法保持不變...
    def compute_roi_region_for_all(self):
        image_ids = list(map(int, self.db.get_image_ids()))
        sorted_image_ids = sorted(image_ids)
        for image_id in sorted_image_ids:
            get_class_id = self.db.get_class_ids_by_image_id(image_id)
            class_ids = list(map(int, get_class_id.keys()))
            for class_id in class_ids:
                self.compute_roi_region_by_ids(image_id, class_id)

    def get_context_awareness(self, image_id, class_id, point1, point2):
        return self._get_roi_region_specific(image_id, class_id, point1, point2)

    def get_roi_region(self):
        image_ids = list(map(int, self.db.get_image_ids()))
        sorted_image_ids = sorted(image_ids)
        roi_regions = []
        for image_id in sorted_image_ids:
            get_class_id = self.db.get_class_ids_by_image_id(image_id)
            class_ids = list(map(int, get_class_id.keys()))
            for class_id in class_ids:
                roi_region = self._get_roi_region_by_ids(image_id, class_id)
                roi_regions.append(roi_region)
        return roi_regions
    
    def _get_roi_region_by_ids(self, image_id, class_id):
        c_x1s = self.db.get_specific_data(image_id, class_id, "context_roi_point1_x")
        c_y1s = self.db.get_specific_data(image_id, class_id, "context_roi_point1_y")
        c_x2s = self.db.get_specific_data(image_id, class_id, "context_roi_point2_x")
        c_y2s = self.db.get_specific_data(image_id, class_id, "context_roi_point2_y")

        context_roi = []
        for cx1, cy1, cx2, cy2 in zip(c_x1s, c_y1s, c_x2s, c_y2s):
            point1 = (float(cx1), float(cy1))
            point2 = (float(cx2), float(cy2))
            context_roi.append((point1, point2))

        return context_roi

    def _get_roi_region_specific(self, image_id, class_id, point1, point2):
        t_x1s = self.db.get_specific_data(image_id, class_id, "point1_x")
        t_y1s = self.db.get_specific_data(image_id, class_id, "point1_y")
        t_x2s = self.db.get_specific_data(image_id, class_id, "point2_x")
        t_y2s = self.db.get_specific_data(image_id, class_id, "point2_y")
        
        c_x1s = self.db.get_specific_data(image_id, class_id, "context_roi_point1_x")
        c_y1s = self.db.get_specific_data(image_id, class_id, "context_roi_point1_y")
        c_x2s = self.db.get_specific_data(image_id, class_id, "context_roi_point2_x")
        c_y2s = self.db.get_specific_data(image_id, class_id, "context_roi_point2_y")

        for tx1, ty1, tx2, ty2, cx1, cy1, cx2, cy2 in zip(t_x1s, t_y1s, t_x2s, t_y2s, c_x1s, c_y1s, c_x2s, c_y2s):
            if cx1 == -1 and cy1 == -1 and cx2 == -1 and cy2 == -1:
                continue
            if (abs(point1[0] - float(tx1)) < 1e-5 and abs(point1[1] - float(ty1)) < 1e-5 and 
                abs(point2[0] - float(tx2)) < 1e-5 and abs(point2[1] - float(ty2)) < 1e-5):
                context_roi_point1 = (float(cx1), float(cy1))
                context_roi_point2 = (float(cx2), float(cy2))
                return context_roi_point1, context_roi_point2
        
        return None, None
