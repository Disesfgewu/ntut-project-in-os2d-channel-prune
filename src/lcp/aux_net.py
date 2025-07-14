import torch
import torch.nn.functional as F

class AuxiliaryNetwork:
    def __init__(self, context_roi_align, db):
        self._context_roi_align = context_roi_align
        self._db = db

    def get_contextual_roi_align(self):
        return self._context_roi_align
    
    def get_db(self):
        return self._db

    def giou(self, image_id, class_id):
        """
        計算指定 image_id 和 class_id 的所有 box 對的 GIoU 值
        
        Args:
            image_id: 圖像ID
            class_id: 類別ID
            
        Returns:
            list: 所有 box 對的 GIoU 值列表
        """
        giou_values = []
        
        # 從資料庫獲取所有相關的 truth box 和 default box 座標
        truth_x1s = self._db.get_specific_data(image_id, class_id, "point1_x")
        truth_y1s = self._db.get_specific_data(image_id, class_id, "point1_y")
        truth_x2s = self._db.get_specific_data(image_id, class_id, "point2_x")
        truth_y2s = self._db.get_specific_data(image_id, class_id, "point2_y")
        
        default_x1s = self._db.get_specific_data(image_id, class_id, "detection_point1_x")
        default_y1s = self._db.get_specific_data(image_id, class_id, "detection_point1_y")
        default_x2s = self._db.get_specific_data(image_id, class_id, "detection_point2_x")
        default_y2s = self._db.get_specific_data(image_id, class_id, "detection_point2_y")
        
        # 檢查是否有有效的檢測資料
        if not default_x1s or len(default_x1s) == 0:
            print(f"No detection data found for image_id={image_id}, class_id={class_id}")
            return []
        
        # 逐一計算每組 box 對的 GIoU
        for i in range(len(truth_x1s)):
            try:
                # 構建 truth box (x1, y1, x2, y2)
                truth_box = (
                    float(truth_x1s[i]), 
                    float(truth_y1s[i]),
                    float(truth_x2s[i]), 
                    float(truth_y2s[i])
                )
                
                # 構建 default box (x1, y1, x2, y2)
                default_box = (
                    float(default_x1s[i]), 
                    float(default_y1s[i]),
                    float(default_x2s[i]), 
                    float(default_y2s[i])
                )
                
                # 計算 GIoU
                giou_value = self._compute_giou(truth_box, default_box)
                giou_values.append(giou_value)
                
            except (ValueError, IndexError) as e:
                print(f"Error processing box pair {i} for image_id={image_id}, class_id={class_id}: {e}")
                continue
        
        return giou_values
    
    def _compute_giou(self, box1, box2):
        """支援梯度的 GIoU 計算"""
        import torch
        
        # 轉換為 tensor（如果還不是的話）
        if not isinstance(box1, torch.Tensor):
            box1 = torch.tensor(box1, dtype=torch.float32, requires_grad=True)
        if not isinstance(box2, torch.Tensor):
            box2 = torch.tensor(box2, dtype=torch.float32, requires_grad=True)
        
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 使用 torch 操作替代 Python 原生操作
        inter_x1 = torch.max(x1_1, x1_2)
        inter_y1 = torch.max(y1_1, y1_2)
        inter_x2 = torch.min(x2_1, x2_2)
        inter_y2 = torch.min(y2_1, y2_2)
        
        # 交集面積
        inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_height = torch.clamp(inter_y2 - inter_y1, min=0)
        intersection_area = inter_width * inter_height
        
        # 各自面積
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # 聯集面積
        union_area = area1 + area2 - intersection_area
        
        # IoU
        iou = intersection_area / (union_area + 1e-7)  # 避免除零
        
        # 最小包圍框
        c_x1 = torch.min(x1_1, x1_2)
        c_y1 = torch.min(y1_1, y1_2)
        c_x2 = torch.max(x2_1, x2_2)
        c_y2 = torch.max(y2_1, y2_2)
        
        c_area = (c_x2 - c_x1) * (c_y2 - c_y1)
        
        # GIoU
        giou = iou - (c_area - union_area) / (c_area + 1e-7)
        
        return giou

    
    def ac_loss(self, image_id, class_id, point1, point2):
        """
        公式 8：計算分類損失 L_ac = Σ E_i
        
        Args:
            image_id: 圖像ID
            class_id: 類別ID
            point1, point2: 真實邊界框座標
            
        Returns:
            torch.Tensor: 分類損失值
        """
        # 1. 獲得 contextual RoIAlign 特徵
        contextual_features = self._context_roi_align.extract_roi_features_specific(
            image_id, class_id, point1, point2
        )
        
        if not contextual_features:
            return torch.tensor(0.0)
        
        # 2. 計算分類損失
        classification_loss = self._compute_ac_loss(contextual_features)
        
        return classification_loss

    def _compute_ac_loss(self, contextual_features):
        """
        內部實作：計算分類損失 L_ac = Σ E_i
        
        Args:
            contextual_features: 從 contextual RoIAlign 獲得的特徵列表
            
        Returns:
            torch.Tensor: 總分類損失
        """
        
        # 確定設備（根據第一個特徵的設備）
        if contextual_features:
            device = contextual_features[0]['features']['combined_feature'].device
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 在正確的設備上初始化損失
        total_classification_loss = torch.tensor(0.0, device=device)
        
        # 遍歷每個特徵進行分類損失計算
        for i, feature_data in enumerate(contextual_features):
            # 提取 combined_feature 並確保在正確設備上
            combined_feature = feature_data['features']['combined_feature'].to(device)
            
            # 將特徵轉換為分類分數
            C, H, W = combined_feature.shape
            
            # 全域平均池化：將空間維度壓縮
            pooled_feature = torch.mean(combined_feature.view(C, -1), dim=1)  # [C]
            
            # 進一步壓縮為單一分類分數
            class_score = torch.mean(pooled_feature)  # scalar
            
            # 使用 sigmoid 轉換為概率
            class_prob = torch.sigmoid(class_score)
            
            # 確保概率在有效範圍內，避免數值問題
            class_prob = torch.clamp(class_prob, min=1e-7, max=1-1e-7)
            
            # 計算交叉熵損失 E_i
            # 在同一設備上創建目標張量
            target = torch.tensor(1.0, device=device)
            
            # 二元交叉熵損失
            cross_entropy_loss = F.binary_cross_entropy(class_prob, target)
            
            # 累加到總損失（現在都在同一設備上）
            total_classification_loss += cross_entropy_loss
        
        return total_classification_loss

    def ar_loss(self, image_id, class_id, point1, point2, m=5):
        """
        公式 9：計算回歸損失 L_ar = Σ (m - G_i)
        
        Args:
            image_id: 圖像ID
            class_id: 類別ID
            point1, point2: 真實邊界框座標
            m: 常數係數 (論文中設為 50)
            
        Returns:
            torch.Tensor: 回歸損失值
        """
        # 1. 獲取靜態的 truth box 和 default box
        truth_boxes, default_boxes = self._get_static_boxes(image_id, class_id, point1, point2)
        
        if len(truth_boxes) == 0 or len(default_boxes) == 0:
            return torch.tensor(0.0)
        
        # 2. 計算回歸損失
        regression_loss = self._compute_ar_loss(truth_boxes, default_boxes, m)
        
        return regression_loss

    def _compute_ar_loss(self, truth_boxes, default_boxes, m=3):
        """支援梯度的回歸損失計算"""
        import torch
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        total_regression_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        for truth_box, default_box in zip(truth_boxes, default_boxes):
            # 轉換為 tensor
            truth_tensor = torch.tensor(truth_box, dtype=torch.float32, device=device, requires_grad=True)
            default_tensor = torch.tensor(default_box, dtype=torch.float32, device=device, requires_grad=True)
            
            # 計算 GIoU（現在支援梯度）
            giou_value = self._compute_giou(truth_tensor, default_tensor)
            
            # 回歸損失
            loss_i = m - giou_value
            total_regression_loss = total_regression_loss + loss_i
        
        return total_regression_loss

        return total_regression_loss

    def _get_static_boxes(self, image_id, class_id, point1, point2):
        """
        獲取靜態的 truth box 和 default box
        這些 box 在第一次檢測後就固定了
        
        Args:
            image_id: 圖像ID
            class_id: 類別ID
            point1, point2: 指定的真實邊界框座標
            
        Returns:
            tuple: (truth_boxes, default_boxes)
        """
        # 從資料庫獲取真實框座標
        truth_x1s = self._db.get_specific_data(image_id, class_id, "point1_x")
        truth_y1s = self._db.get_specific_data(image_id, class_id, "point1_y")
        truth_x2s = self._db.get_specific_data(image_id, class_id, "point2_x")
        truth_y2s = self._db.get_specific_data(image_id, class_id, "point2_y")
        
        # 從資料庫獲取檢測框座標（來自原始網路，靜態的）
        detect_x1s = self._db.get_specific_data(image_id, class_id, "detection_point1_x")
        detect_y1s = self._db.get_specific_data(image_id, class_id, "detection_point1_y")
        detect_x2s = self._db.get_specific_data(image_id, class_id, "detection_point2_x")
        detect_y2s = self._db.get_specific_data(image_id, class_id, "detection_point2_y")
        
        if not detect_x1s or len(detect_x1s) == 0:
            print(f"No detection data found for image_id={image_id}, class_id={class_id}")
            return [], []
        
        truth_boxes = []
        default_boxes = []
        
        # 只處理匹配指定 point1, point2 的資料
        for i in range(len(truth_x1s)):
            if (abs(float(truth_x1s[i]) - point1[0]) < 1e-5 and 
                abs(float(truth_y1s[i]) - point1[1]) < 1e-5 and
                abs(float(truth_x2s[i]) - point2[0]) < 1e-5 and 
                abs(float(truth_y2s[i]) - point2[1]) < 1e-5):
                
                # 構建 truth box (x1, y1, x2, y2)
                truth_box = (
                    float(truth_x1s[i]), float(truth_y1s[i]),
                    float(truth_x2s[i]), float(truth_y2s[i])
                )
                
                # 構建 default box (x1, y1, x2, y2)
                default_box = (
                    float(detect_x1s[i]), float(detect_y1s[i]),
                    float(detect_x2s[i]), float(detect_y2s[i])
                )
                
                truth_boxes.append(truth_box)
                default_boxes.append(default_box)
        
        return truth_boxes, default_boxes


    def aux_loss(self, image_id, class_id, point1, point2):
        return self.ar_loss(image_id, class_id, point1, point2) + self.ac_loss(image_id, class_id, point1, point2)
    
    def approximate_ar_loss_no_grad(self, image_id, class_id, point1, point2, m=3):
        """
        基於數學推導的回歸損失近似
        實現公式：S_k^ar = |μ_k - μ_global|/σ_global × GIoU_penalty
        """
        try:
            # 1. 獲取 Context RoI 特徵（使用您的無梯度 API）
            context_roi_point1, context_roi_point2 = self._context_roi_align._get_roi_region_specific(
                image_id, class_id, point1, point2
            )
            
            if context_roi_point1 is None or context_roi_point2 is None:
                return 0.0
                
            contextual_features = self._context_roi_align.extract_roi_align_no_grad(
                image_id, class_id, point1, point2,
                context_roi_point1, context_roi_point2
            )
            
            if contextual_features is None:
                return 0.0
            
            # 2. 實現數學公式：定位敏感性計算
            combined_feature = contextual_features['combined_feature']
            C, H, W = combined_feature.shape
            
            # 全域統計
            global_mean = torch.mean(combined_feature)
            global_std = torch.std(combined_feature) + 1e-8
            
            # 逐通道均值偏差（理論公式核心）
            channel_means = torch.mean(combined_feature.view(C, -1), dim=1)
            mean_deviations = torch.abs(channel_means - global_mean)
            localization_sensitivity = torch.mean(mean_deviations) / global_std
            
            # 3. GIoU 懲罰項計算
            giou_stats = self._compute_giou_statistics(image_id, class_id)
            giou_penalty = max(0, m - giou_stats['mean']) if giou_stats['count'] > 0 else 0
            
            # 4. 應用理論公式
            ar_loss_approx = localization_sensitivity.item() * giou_penalty
            
            return float(ar_loss_approx)
            
        except Exception as e:
            print(f"[ERROR] 回歸損失近似失敗: {e}")
            return 0.0

    def _compute_giou_statistics(self, image_id, class_id):
        """計算 GIoU 統計特性"""
        try:
            giou_values = self.giou(image_id, class_id)
            if not giou_values:
                return {'mean': 0.0, 'std': 0.0, 'count': 0}
            
            import numpy as np
            giou_array = np.array(giou_values)
            return {
                'mean': float(np.mean(giou_array)),
                'std': float(np.std(giou_array)),
                'count': len(giou_values)
            }
        except:
            return {'mean': 0.0, 'std': 0.0, 'count': 0}
        
    def approximate_ac_loss_no_grad(self, image_id, class_id, point1, point2):
        """
        基於數學推導的分類損失近似
        實現公式：S_k^ac = Var(F_k)/σ_global² × CE_avg
        """
        try:
            # 獲取 contextual 特徵
            context_roi_point1, context_roi_point2 = self._context_roi_align._get_roi_region_specific(
                image_id, class_id, point1, point2
            )
            
            if context_roi_point1 is None or context_roi_point2 is None:
                return 0.0
                
            contextual_features = self._context_roi_align.extract_roi_align_no_grad(
                image_id, class_id, point1, point2,
                context_roi_point1, context_roi_point2
            )
            
            if contextual_features is None:
                return 0.0
            
            # 實現理論公式：方差分析
            combined_feature = contextual_features['combined_feature']
            
            # 全域統計
            global_std = torch.std(combined_feature) + 1e-8
            
            # 通道方差計算（判別能力指標）
            C, H, W = combined_feature.shape
            channel_variances = torch.var(combined_feature, dim=(1, 2))
            avg_variance = torch.mean(channel_variances)
            
            # 歸一化方差分數（理論公式）
            variance_score = avg_variance / (global_std ** 2)
            
            # 分類敏感性評估
            classification_sensitivity = self._estimate_classification_sensitivity(contextual_features)
            
            # 整合分類損失近似
            ac_loss_approx = 0.6 * variance_score.item() + 0.4 * classification_sensitivity
            
            return float(ac_loss_approx)
            
        except Exception as e:
            print(f"[ERROR] 分類損失近似失敗: {e}")
            return 0.0

    def _estimate_classification_sensitivity(self, contextual_features):
        """基於特徵差異性的分類敏感性評估"""
        truth_feature = contextual_features['truth_feature']
        context_feature = contextual_features['context_feature']
        
        # 計算特徵間差異（判別能力指標）
        feature_diff = torch.abs(truth_feature - context_feature)
        diff_energy = torch.mean(feature_diff)
        
        # 歸一化敏感性
        combined_std = torch.std(contextual_features['combined_feature']) + 1e-8
        sensitivity = diff_energy / combined_std
        
        return sensitivity.item()
