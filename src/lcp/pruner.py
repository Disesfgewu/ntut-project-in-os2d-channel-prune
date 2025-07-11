import torch

class Pruner:
    def __init__(self, prune_network):
        self.prune_network = prune_network
        self.prune_db = None

    def set_prune_db(self, prune_db):
        self.prune_db = prune_db

    def get_prune_layers_keep_indices(self, layer_name):
        return self.prune_db.get_layer_keep_indices(layer_name)
    
    def prune_layer(self, layer_name):
        """
        主要剪枝函數 - 協調所有剪枝步驟
        """
        print(f"[PRUNE] 開始剪枝層級: {layer_name}")
        
        # 1. 獲取保留索引
        if layer_name.startswith('net_feature_maps.'):
            keep_indices = self.get_prune_layers_keep_indices(layer_name)
        else:
            keep_indices = self.get_prune_layers_keep_indices(f"net_feature_maps.{layer_name}")    
        if not keep_indices:
            print(f"[WARNING] {layer_name} 無保留索引，跳過剪枝")
            return
        
        # 2. 解析層級依賴關係
        dependencies = self.resolve_layer_dependencies(layer_name)
        
        # 3. 剪枝當前層的輸出通道
        self._prune_out_channel(layer_name, keep_indices)
        
        # 4. 剪枝對應的 BatchNorm 層
        if dependencies.get('batch_norm'):
            self._prune_batchnorm_layer(dependencies['batch_norm'], keep_indices)

        # 5. 剪枝下游層的輸入通道
        if dependencies.get('downstream_layers'):
            for next_layer in dependencies['downstream_layers']:
                self._prune_next_layer_inputs(layer_name, next_layer, keep_indices)
        
        # 6. 處理殘差連接
        if dependencies.get('downsample_layer'):
            check , downsample_layer_name = self._should_process_downsample(layer_name, dependencies)
            if check:
                self._prune_downsample_connection(downsample_layer_name, keep_indices)

        # 8. 更新通道索引追蹤
        self.track_channel_indices(layer_name, keep_indices)
    
        print(f"[PRUNE] 完成剪枝層級: {layer_name}")
    
    def _should_process_downsample(self, layer_name, dependencies):
        """
        判斷是否需要處理 downsample 層
        
        Args:
            layer_name: 當前剪枝的層名稱
            dependencies: 依賴關係字典
            
        Returns:
            tuple: (should_process, downsample_path)
        """
        # 解析層名稱
        clean_name = layer_name.replace('net_feature_maps.', '')
        parts = clean_name.split('.')
        
        # 只有滿足以下條件才需要處理 downsample：
        # 1. 是 conv3 層
        # 2. 是第一個 block (index 0)
        # 3. 實際存在 downsample 層
        
        if (len(parts) >= 3 and 
            parts[2] == 'conv3' and 
            parts[1] == '0'):
            
            # 構建 downsample 路徑
            downsample_path = f"net_feature_maps.{parts[0]}.0.downsample"
            
            # 檢查是否真的存在
            if self._verify_layer_exists(downsample_path):
                return True, downsample_path
        
        return False, None

    def _prune_next_layer_inputs(
            self,
            current_layer_name: str,
            next_layer_name: str,
            keep_indices: list
        ) -> bool:
        """
        同步裁剪下一層的輸入通道 / 特徵

        Args
        ----
        current_layer_name : 當前已剪枝層 (僅用於日誌)
        next_layer_name    : 需修補的下一層 (Conv / Linear)
        keep_indices       : 前一層保留的輸出通道索引

        Returns
        -------
        bool : 剪枝是否成功
        """
        try:
            print(f"[PRUNE] 修補 {next_layer_name} 的輸入通道 (來源 {current_layer_name})")

            next_layer = self._get_layer_by_path(next_layer_name)
            if next_layer is None:
                print(f"[WARNING] 找不到層：{next_layer_name}，跳過修補")
                return False

            # ---------- 1. Conv 系列 ----------
            if isinstance(next_layer, (torch.nn.Conv1d,
                                    torch.nn.Conv2d,
                                    torch.nn.Conv3d)):
                with torch.no_grad():
                    w = next_layer.weight.data
                    # w 形狀: [out_c, in_c, ...]
                    new_w = w[:, keep_indices, ...].clone()
                    next_layer.weight = torch.nn.Parameter(new_w)
                    next_layer.in_channels = len(keep_indices)

                print(f"[PRUNE] Conv 輸入維度: {w.shape} → {new_w.shape}")
                return True

            # ---------- 2. 線性層 ----------
            if isinstance(next_layer, torch.nn.Linear):
                with torch.no_grad():
                    w = next_layer.weight.data          # [out_f, in_f]
                    new_w = w[:, keep_indices].clone()
                    next_layer.weight = torch.nn.Parameter(new_w)
                    next_layer.in_features = len(keep_indices)

                print(f"[PRUNE] Linear 輸入維度: {w.shape} → {new_w.shape}")
                return True

            # ---------- 3. GroupNorm / LayerNorm (可選處理) ----------
            if isinstance(next_layer, torch.nn.GroupNorm):
                # GroupNorm 的 weight / bias 與 in_channels 等長
                with torch.no_grad():
                    next_layer.weight = torch.nn.Parameter(
                        next_layer.weight[keep_indices].clone())
                    next_layer.bias = torch.nn.Parameter(
                        next_layer.bias[keep_indices].clone())
                next_layer.num_channels = len(keep_indices)
                print(f"[PRUNE] GroupNorm 通道同步完成")
                return True

            # 其他層型別暫不處理
            print(f"[INFO] {next_layer_name} 類型 {type(next_layer)} 無需修補")
            return True

        except Exception as e:
            print(f"[ERROR] 修補 {next_layer_name} 失敗：{e}")
            import traceback
            traceback.print_exc()
            return False

    def _prune_in_channel_for_connection_fix(
        self,
        layer_name: str,
        keep_indices: list[int]
    ) -> bool:
        """
        修正 *當前層* 的輸入通道，通常只有下列兩種情況才會呼叫：
        1. 第一個 block 的 conv1（因上一個 stage downsample）
        2. 殘差分支中的 downsample.conv

        Args
        ----
        layer_name   : 需要修補的層（卷積 / 線性 / 歸一化）
        keep_indices : 來源層保留下來的 out-channel 索引

        Returns
        -------
        bool : 是否修補成功
        """
        try:
            layer = self._get_layer_by_path(layer_name)
            if layer is None:
                print(f"[WARN] 找不到層：{layer_name}，跳過 in-channel 修補")
                return False

            # ---------------- 1. Conv 層 ----------------
            if isinstance(layer, (torch.nn.Conv1d,
                                torch.nn.Conv2d,
                                torch.nn.Conv3d)):

                if layer.in_channels == len(keep_indices):
                    # 通道數未變化，直接返回
                    return True  

                with torch.no_grad():
                    w_old = layer.weight.data          # [out_c, in_c, ...]
                    w_new = w_old[:, keep_indices, ...].clone()
                    layer.weight  = torch.nn.Parameter(w_new)
                    layer.in_channels = len(keep_indices)

                print(f"[PRUNE] Conv 修補 {layer_name}: "
                    f"in {w_old.shape} ➜ {w_new.shape}")
                return True

            # ---------------- 2. 線性層 ----------------
            if isinstance(layer, torch.nn.Linear):
                if layer.in_features == len(keep_indices):
                    return True

                with torch.no_grad():
                    w_old = layer.weight.data          # [out_f, in_f]
                    w_new = w_old[:, keep_indices].clone()
                    layer.weight = torch.nn.Parameter(w_new)
                    layer.in_features = len(keep_indices)

                print(f"[PRUNE] Linear 修補 {layer_name}: "
                    f"in {w_old.shape} ➜ {w_new.shape}")
                return True

            # ---------------- 3. (Group)Norm 層 ----------------
            if isinstance(layer, (torch.nn.BatchNorm1d,
                                torch.nn.BatchNorm2d,
                                torch.nn.GroupNorm,
                                torch.nn.LayerNorm)):

                if layer.weight.shape[0] == len(keep_indices):
                    return True

                with torch.no_grad():
                    if hasattr(layer, "weight") and layer.weight is not None:
                        layer.weight = torch.nn.Parameter(
                            layer.weight[keep_indices].clone())
                    if hasattr(layer, "bias") and layer.bias is not None:
                        layer.bias   = torch.nn.Parameter(
                            layer.bias[keep_indices].clone())

                    # running_* 僅 BatchNorm 具有
                    if hasattr(layer, "running_mean"):
                        layer.running_mean = layer.running_mean[keep_indices].clone()
                    if hasattr(layer, "running_var"):
                        layer.running_var  = layer.running_var[keep_indices].clone()

                # num_features / num_channels / normalized_shape
                if hasattr(layer, "num_features"):
                    layer.num_features = len(keep_indices)
                if hasattr(layer, "num_channels"):
                    layer.num_channels = len(keep_indices)
                if hasattr(layer, "normalized_shape"):
                    layer.normalized_shape = (len(keep_indices),)

                print(f"[PRUNE] Norm 修補 {layer_name}: "
                    f"features ➜ {len(keep_indices)}")
                return True

            # ---------------------------------------------------
            print(f"[INFO] {layer_name} 類型 {type(layer)} 無需修補")
            return True

        except Exception as e:
            print(f"[ERROR] 修補 {layer_name} 失敗：{e}")
            import traceback; traceback.print_exc()
            return False

    def _prune_batchnorm_layer(self, bn_layer_name, keep_indices):
        """
        剪枝 BatchNorm 層的所有參數
        
        Args:
            bn_layer_name: BatchNorm 層的名稱（如 'net_feature_maps.layer2.1.bn2'）
            keep_indices: 要保留的通道索引列表
        """
        try:
            print(f"[PRUNE] 開始剪枝 BatchNorm 層: {bn_layer_name}")
            
            # 1. 獲取 BatchNorm 層對象
            bn_layer = self._get_layer_by_name(bn_layer_name)
            
            if bn_layer is None:
                print(f"[WARNING] 找不到 BatchNorm 層: {bn_layer_name}")
                return False
            
            # 2. 驗證層類型
            if not isinstance(bn_layer, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                print(f"[WARNING] {bn_layer_name} 不是 BatchNorm 層，類型: {type(bn_layer)}")
                return False
            
            # 3. 記錄原始參數信息
            original_features = bn_layer.num_features
            new_features = len(keep_indices)
            
            print(f"[PRUNE] BatchNorm 通道數變化: {original_features} -> {new_features}")
            
            # 4. 剪枝可學習參數
            with torch.no_grad():
                # 剪枝 weight (gamma 參數)
                if hasattr(bn_layer, 'weight') and bn_layer.weight is not None:
                    new_weight = bn_layer.weight[keep_indices].clone()
                    bn_layer.weight = torch.nn.Parameter(new_weight)
                    print(f"[PRUNE] BatchNorm weight 剪枝完成: {new_weight.shape}")
                
                # 剪枝 bias (beta 參數)
                if hasattr(bn_layer, 'bias') and bn_layer.bias is not None:
                    new_bias = bn_layer.bias[keep_indices].clone()
                    bn_layer.bias = torch.nn.Parameter(new_bias)
                    print(f"[PRUNE] BatchNorm bias 剪枝完成: {new_bias.shape}")
            
            # 5. 剪枝運行時統計參數
            if hasattr(bn_layer, 'running_mean') and bn_layer.running_mean is not None:
                bn_layer.running_mean = bn_layer.running_mean[keep_indices].clone()
                print(f"[PRUNE] BatchNorm running_mean 剪枝完成: {bn_layer.running_mean.shape}")
            
            if hasattr(bn_layer, 'running_var') and bn_layer.running_var is not None:
                bn_layer.running_var = bn_layer.running_var[keep_indices].clone()
                print(f"[PRUNE] BatchNorm running_var 剪枝完成: {bn_layer.running_var.shape}")
            
            # 6. 更新 num_features 屬性
            bn_layer.num_features = new_features
            
            # 7. 驗證剪枝結果
            self._validate_bn_pruning(bn_layer, keep_indices, bn_layer_name)
            
            print(f"[PRUNE] BatchNorm 層 {bn_layer_name} 剪枝成功")
            return True
            
        except Exception as e:
            print(f"[ERROR] 剪枝 BatchNorm 層 {bn_layer_name} 失敗: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_layer_by_name(self, layer_name):
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
            
            # 從剪枝網路開始遍歷
            layer = self.prune_network.net_feature_maps
            
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

    def _validate_bn_pruning(self, bn_layer, keep_indices, bn_layer_name):
        """
        驗證 BatchNorm 剪枝結果的正確性
        
        Args:
            bn_layer: 剪枝後的 BatchNorm 層
            keep_indices: 保留的通道索引
            bn_layer_name: 層名稱
        """
        try:
            expected_features = len(keep_indices)
            
            # 檢查 num_features
            assert bn_layer.num_features == expected_features, \
                f"num_features 不匹配: {bn_layer.num_features} vs {expected_features}"
            
            # 檢查參數維度
            if bn_layer.weight is not None:
                assert bn_layer.weight.shape[0] == expected_features, \
                    f"weight 維度不匹配: {bn_layer.weight.shape[0]} vs {expected_features}"
            
            if bn_layer.bias is not None:
                assert bn_layer.bias.shape[0] == expected_features, \
                    f"bias 維度不匹配: {bn_layer.bias.shape[0]} vs {expected_features}"
            
            if bn_layer.running_mean is not None:
                assert bn_layer.running_mean.shape[0] == expected_features, \
                    f"running_mean 維度不匹配: {bn_layer.running_mean.shape[0]} vs {expected_features}"
            
            if bn_layer.running_var is not None:
                assert bn_layer.running_var.shape[0] == expected_features, \
                    f"running_var 維度不匹配: {bn_layer.running_var.shape[0]} vs {expected_features}"
            
            print(f"[VALIDATE] BatchNorm 層 {bn_layer_name} 剪枝驗證通過")
            
        except AssertionError as e:
            print(f"[ERROR] BatchNorm 剪枝驗證失敗 {bn_layer_name}: {e}")
            raise
        except Exception as e:
            print(f"[ERROR] BatchNorm 剪枝驗證出錯 {bn_layer_name}: {e}")
            raise

    def _prune_downsample_connection(self, downsample_layer_path: str,
                                 keep_indices: list[int]) -> None:
        """
        裁剪 ResNet downsample Sequential 之 conv + bn。

        Args
        ----
        downsample_layer_path : 例 "net_feature_maps.layer2.0.downsample"
        keep_indices          : 保留的 out-channel 索引
        """
        if not keep_indices:
            print(f"[WARN] keep_indices 為空，跳過 downsample 剪枝")
            return

        if not self._verify_layer_exists(downsample_layer_path):
            print(f"[WARN] 找不到 downsample 層：{downsample_layer_path}")
            return

        seq_layer = self._get_layer_by_path(downsample_layer_path)
        if not isinstance(seq_layer, torch.nn.Sequential):
            print(f"[WARN] {downsample_layer_path} 非 Sequential，跳過")
            return

        # 依序期望：0→Conv2d，1→BatchNorm2d
        conv_layer: torch.nn.Conv2d  = seq_layer[0]
        bn_layer:   torch.nn.Module  = seq_layer[1]

        ori_device = conv_layer.weight.device  # 記錄原始 device
        conv_layer.cpu();  bn_layer.cpu()      # 先搬到 CPU，避免 CUDA assert

        try:
            # ---------- Conv 剪枝 ----------
            out_c, in_c, k_h, k_w = conv_layer.weight.shape
            assert max(keep_indices) < out_c, \
                f"索引超界：max={max(keep_indices)}, out_c={out_c}"

            conv_layer.weight = torch.nn.Parameter(
                conv_layer.weight[keep_indices].clone()
            )
            if conv_layer.bias is not None:
                conv_layer.bias = torch.nn.Parameter(
                    conv_layer.bias[keep_indices].clone()
                )
            conv_layer.out_channels = len(keep_indices)

            # groups 需能整除 in/out，常見 downsample.groups==1
            if conv_layer.groups > 1:
                conv_layer.groups = min(conv_layer.groups, len(keep_indices))

            # ---------- BN 剪枝 ----------
            num_feat = bn_layer.num_features
            assert max(keep_indices) < num_feat, \
                f"BN 索引超界：max={max(keep_indices)}, feat={num_feat}"

            bn_layer.weight = torch.nn.Parameter(
                bn_layer.weight[keep_indices].clone()
            )
            bn_layer.bias   = torch.nn.Parameter(
                bn_layer.bias[keep_indices].clone()
            )
            bn_layer.running_mean = bn_layer.running_mean[keep_indices].clone()
            bn_layer.running_var  = bn_layer.running_var[keep_indices].clone()
            bn_layer.num_features = len(keep_indices)

            print(f"[PRUNE✔] {downsample_layer_path} "
                f"Conv/BN : {out_c} ➜ {len(keep_indices)}")

        except Exception as e:
            print(f"[PRUNE✘] Downsample 剪枝失敗 ({downsample_layer_path}): {e}")
            raise

        finally:
            # 恢復到原 device
            seq_layer.to(ori_device)


    def _prune_conv_in_downsample(self, conv_layer, layer_path, keep_indices):
        """剪枝 downsample 中的卷積層"""
        try:
            original_channels = conv_layer.out_channels
            
            with torch.no_grad():
                # 剪枝輸出通道
                new_weight = conv_layer.weight[keep_indices].clone()
                conv_layer.weight = torch.nn.Parameter(new_weight)
                
                if conv_layer.bias is not None:
                    new_bias = conv_layer.bias[keep_indices].clone()
                    conv_layer.bias = torch.nn.Parameter(new_bias)
                
                conv_layer.out_channels = len(keep_indices)
            
            print(f"[Pruner] Downsample Conv {layer_path}: {original_channels} → {len(keep_indices)}")
            
        except Exception as e:
            print(f"[ERROR] Downsample Conv 剪枝失敗 {layer_path}: {e}")

    def _prune_bn_in_downsample(self, bn_layer, layer_path, keep_indices):
        """剪枝 downsample 中的 BatchNorm 層"""
        try:
            original_features = bn_layer.num_features
            
            with torch.no_grad():
                if hasattr(bn_layer, 'weight') and bn_layer.weight is not None:
                    bn_layer.weight = torch.nn.Parameter(bn_layer.weight[keep_indices].clone())
                
                if hasattr(bn_layer, 'bias') and bn_layer.bias is not None:
                    bn_layer.bias = torch.nn.Parameter(bn_layer.bias[keep_indices].clone())
                
                if hasattr(bn_layer, 'running_mean'):
                    bn_layer.running_mean = bn_layer.running_mean[keep_indices].clone()
                
                if hasattr(bn_layer, 'running_var'):
                    bn_layer.running_var = bn_layer.running_var[keep_indices].clone()
                
                bn_layer.num_features = len(keep_indices)
            
            print(f"[Pruner] Downsample BN {layer_path}: {original_features} → {len(keep_indices)}")
            
        except Exception as e:
            print(f"[ERROR] Downsample BN 剪枝失敗 {layer_path}: {e}")


    def track_channel_indices(self, layer_name: str, keep_indices: list[int]) -> None:
        """
        將剪枝結果寫回資料庫並保存在記憶體

        欄位：
        - layer                  : 層名稱
        - original_channel_num   : 剪枝前通道數
        - num_of_keep_channel    : 剩餘通道數
        - keep_index             : list 文字串
        """
        # ------- 1. 取得原始通道數 -------
        layer_obj = self._get_layer_by_path(layer_name)
        if layer_obj is None:
            print(f"[WARN] track_channel 無法找到層：{layer_name}")
            return

        ori_channels = self._get_output_channels(layer_obj)

        # ------- 2. 寫回 CSV (若有設定 prune_db) -------
        if self.prune_db is not None:
            try:
                self.prune_db.write_data(layer   = layer_name,
                                        original_channel_num = ori_channels,
                                        num_of_keep_channel  = len(keep_indices),
                                        keep_index           = keep_indices)
            except Exception as e:
                print(f"[WARN] 寫入 prune_db 失敗：{e}")

        # ------- 3. 緩存於記憶體 -------
        if not hasattr(self, "_prune_history"):
            self._prune_history = {}

        self._prune_history[layer_name] = {
            "original_channels": ori_channels,
            "kept_channels"    : len(keep_indices),
            "prune_ratio"      : 1 - len(keep_indices)/ori_channels,
            "keep_indices"     : keep_indices.copy()
        }

        print(f"[TRACK] {layer_name} 剪枝率 "
            f"{self._prune_history[layer_name]['prune_ratio']:.1%}")


    def _prune_out_channel(self, layer_name, keep_indexes):
        """
        剪枝指定層的輸出通道
        
        Args:
            layer_name (str): 層名稱，如 'net_feature_maps.layer3.5.conv3'
            keep_indexes (list): 要保留的通道索引列表
        
        Returns:
            bool: 剪枝是否成功
        """
        try:
            print(f"[PRUNE] 開始剪枝輸出通道: {layer_name}")
            print(f"[PRUNE] 保留通道索引: {keep_indexes}")
            print(f"[PRUNE] 保留通道數量: {len(keep_indexes)}")
            
            # 1. 獲取目標層對象
            target_layer = self._get_layer_by_path(layer_name)
            if target_layer is None:
                print(f"[ERROR] 無法找到層: {layer_name}")
                return False
            
            # 2. 驗證層類型
            if not self._is_prunable_layer(target_layer):
                print(f"[ERROR] 層 {layer_name} 不支援剪枝")
                return False
            
            # 3. 記錄原始資訊
            original_channels = self._get_output_channels(target_layer)
            print(f"[PRUNE] 原始輸出通道數: {original_channels}")
            
            # 4. 驗證索引有效性
            if not self._validate_keep_indexes(keep_indexes, original_channels):
                print(f"[ERROR] 保留索引無效")
                return False
            
            # 5. 執行權重剪枝
            success = self._prune_layer_weights(target_layer, keep_indexes, layer_name)
            if not success:
                return False
            
            # 6. 更新層屬性
            self._update_layer_attributes(target_layer, len(keep_indexes))
            
            print(f"[PRUNE] 輸出通道剪枝完成: {original_channels} -> {len(keep_indexes)}")
            return True
            
        except Exception as e:
            print(f"[ERROR] 剪枝輸出通道失敗 {layer_name}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_layer_by_path(self, layer_path):
        """根據層路徑獲取層對象"""
        try:
            # 移除 'net_feature_maps.' 前綴
            clean_path = layer_path.replace('net_feature_maps.', '')
            
            # 從 prune_network 開始導航
            current_layer = self.prune_network.net_feature_maps
            
            # 按路徑逐級訪問
            for attr in clean_path.split('.'):
                if hasattr(current_layer, attr):
                    current_layer = getattr(current_layer, attr)
                elif attr.isdigit():
                    current_layer = current_layer[int(attr)]
                else:
                    print(f"[ERROR] 路徑 '{attr}' 在 {clean_path} 中不存在")
                    return None
            
            return current_layer
            
        except Exception as e:
            print(f"[ERROR] 獲取層對象失敗: {e}")
            return None

    def _is_prunable_layer(self, layer):
        """檢查層是否可以剪枝"""
        # 檢查是否為卷積層或線性層
        prunable_types = (
            torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
            torch.nn.Linear
        )
        
        if not isinstance(layer, prunable_types):
            return False
        
        # 檢查是否有權重參數
        if not hasattr(layer, 'weight') or layer.weight is None:
            return False
        
        return True

    def _get_output_channels(self, layer):
        """獲取層的輸出通道數"""
        if hasattr(layer, 'out_channels'):
            return layer.out_channels
        elif hasattr(layer, 'out_features'):
            return layer.out_features
        elif hasattr(layer, 'weight'):
            return layer.weight.shape[0]  # 第一維通常是輸出通道
        else:
            return 0

    def _validate_keep_indexes(self, keep_indexes, total_channels):
        """驗證保留索引的有效性"""
        if not keep_indexes:
            print("[ERROR] 保留索引列表為空")
            return False
        
        # 檢查索引範圍
        for idx in keep_indexes:
            if not isinstance(idx, int) or idx < 0 or idx >= total_channels:
                print(f"[ERROR] 索引 {idx} 超出範圍 [0, {total_channels-1}]")
                return False
        
        # 檢查索引唯一性
        if len(set(keep_indexes)) != len(keep_indexes):
            print("[ERROR] 保留索引包含重複值")
            return False
        
        return True

    def _prune_layer_weights(self, layer, keep_indexes, layer_name):
        """執行實際的權重剪枝"""
        try:
            with torch.no_grad():
                # 1. 剪枝權重參數
                original_weight = layer.weight.data
                
                if len(original_weight.shape) == 4:  # Conv2d: [out_channels, in_channels, H, W]
                    new_weight = original_weight[keep_indexes, :, :, :].clone()
                    print(f"[PRUNE] Conv2d 權重: {original_weight.shape} -> {new_weight.shape}")
                    
                elif len(original_weight.shape) == 2:  # Linear: [out_features, in_features]
                    new_weight = original_weight[keep_indexes, :].clone()
                    print(f"[PRUNE] Linear 權重: {original_weight.shape} -> {new_weight.shape}")
                    
                elif len(original_weight.shape) == 3:  # Conv1d: [out_channels, in_channels, kernel_size]
                    new_weight = original_weight[keep_indexes, :, :].clone()
                    print(f"[PRUNE] Conv1d 權重: {original_weight.shape} -> {new_weight.shape}")
                    
                else:
                    print(f"[ERROR] 不支援的權重維度: {original_weight.shape}")
                    return False
                
                # 更新權重參數
                layer.weight = torch.nn.Parameter(new_weight)
                
                # 2. 剪枝偏置參數（如果存在）
                if hasattr(layer, 'bias') and layer.bias is not None:
                    original_bias = layer.bias.data
                    new_bias = original_bias[keep_indexes].clone()
                    layer.bias = torch.nn.Parameter(new_bias)
                    print(f"[PRUNE] 偏置: {original_bias.shape} -> {new_bias.shape}")
                
                return True
                
        except Exception as e:
            print(f"[ERROR] 權重剪枝失敗: {e}")
            return False

    def _update_layer_attributes(self, layer, new_channel_count):
        """更新層的屬性以反映剪枝後的通道數"""
        try:
            # 更新輸出通道數或特徵數
            if hasattr(layer, 'out_channels'):
                layer.out_channels = new_channel_count
                print(f"[PRUNE] 更新 out_channels: {new_channel_count}")
                
            elif hasattr(layer, 'out_features'):
                layer.out_features = new_channel_count
                print(f"[PRUNE] 更新 out_features: {new_channel_count}")
            
            # 對於 GroupNorm 或其他需要通道數的層，可能需要額外處理
            if hasattr(layer, 'groups') and layer.groups > 1:
                # 確保新的通道數能被 groups 整除
                if new_channel_count % layer.groups != 0:
                    print(f"[WARNING] 新通道數 {new_channel_count} 無法被 groups {layer.groups} 整除")
                    # 可以選擇調整 groups 或報錯
            
        except Exception as e:
            print(f"[WARNING] 更新層屬性失敗: {e}")

    def resolve_layer_dependencies(self, target_layer):
        """
        解析目標層級的依賴關係，確定剪枝時需要同步處理的層級
        
        Args:
            target_layer (str): 目標層名稱，如 'net_feature_maps.layer2.1.conv2'
        
        Returns:
            dict: 包含依賴關係的字典
            {
                'target': str,                    # 目標層名稱
                'batch_norm': str or None,        # 對應的 BatchNorm 層
                'downstream_layers': list,        # 下游依賴層級
                'skip_connections': list,         # 跳躍連接相關層級
                'dependency_type': str,           # 依賴類型：'conv', 'residual', 'downsample'
                'pruning_strategy': str           # 建議的剪枝策略
            }
        """
        
        # 初始化返回結果
        dependencies = {
            'target': target_layer,
            'batch_norm': None,
            'downstream_layers': [],
            'downsample_layer': None,
            'skip_connections': [],
            'dependency_type': 'conv',
            'pruning_strategy': 'channel_pruning',
            'needs_input_fix': False
        }
        
        try:
            # 1. 解析層級路徑
            layer_parts = target_layer.replace('net_feature_maps.', '').split('.')
            
            # 2. 確定 BatchNorm 層
            batch_norm_layer = self._find_corresponding_batch_norm(target_layer, layer_parts)
            dependencies['batch_norm'] = batch_norm_layer
            
            # 3. 分析層級類型和位置
            layer_type, block_info = self._analyze_layer_type(layer_parts)
            dependencies['dependency_type'] = layer_type
            
            # 4. 找出下游依賴層級
            downstream_layers = self._find_downstream_layers(target_layer, layer_parts, block_info)
            dependencies['downstream_layers'] = downstream_layers
            
            # 5. 處理 ResNet 跳躍連接和 downsample
            if layer_type == 'residual' and target_layer.endswith('conv3'):
                skip_connections, downsample_layer = self._find_skip_connections_and_downsample(layer_parts, block_info)
                dependencies['skip_connections'] = skip_connections
                dependencies['downsample_layer'] = downsample_layer
            
            # 6. 確定剪枝策略
            pruning_strategy = self._determine_pruning_strategy(layer_type, block_info)
            dependencies['pruning_strategy'] = pruning_strategy
            
            dependencies['needs_input_fix'] = self._needs_input_channel_fix(layer_parts, block_info)

            print(f"[LCP] 層級依賴分析完成 - {target_layer}")
            print(f"      BatchNorm: {batch_norm_layer}")
            print(f"      下游層級: {len(downstream_layers)} 個")
            print(f"      跳躍連接: {len(dependencies['skip_connections'])} 個")
            
            return dependencies
            
        except Exception as e:
            print(f"[ERROR] 層級依賴分析失敗: {e}")
            return dependencies

    def _find_corresponding_batch_norm(self, target_layer, layer_parts):
        """找出對應的 BatchNorm 層"""
        
        # 常見的 conv -> bn 對應關係
        conv_to_bn_mapping = {
            'conv1': 'bn1',
            'conv2': 'bn2', 
            'conv3': 'bn3'
        }
        
        # 檢查是否為卷積層
        if layer_parts[-1] in conv_to_bn_mapping:
            bn_name = conv_to_bn_mapping[layer_parts[-1]]
            
            # 構建 BatchNorm 層路徑
            bn_parts = layer_parts[:-1] + [bn_name]
            bn_layer = 'net_feature_maps.' + '.'.join(bn_parts)
            
            # 驗證 BatchNorm 層是否存在
            if self._verify_layer_exists(bn_layer):
                return bn_layer
        
        return None

    def _analyze_layer_type(self, layer_parts):
        """分析層級類型和位置信息"""
        
        if len(layer_parts) == 1:
            # 如 conv1
            return 'standalone', {'position': 'entry'}
        
        elif len(layer_parts) >= 3:
            # 如 layer1.0.conv1, layer2.1.conv2
            layer_group = layer_parts[0]  # layer1, layer2, etc.
            block_idx = int(layer_parts[1])  # 0, 1, 2, etc.
            conv_name = layer_parts[2]  # conv1, conv2, conv3
            
            block_info = {
                'group': layer_group,
                'block_index': block_idx,
                'conv_position': conv_name,
                'is_first_block': block_idx == 0,
                'is_downsample_layer': block_idx == 0 and layer_group != 'layer1'
            }
            
            return 'residual', block_info
        
        return 'unknown', {}

    def _find_downstream_layers(self, target_layer, layer_parts, block_info):
        """找出下游依賴層級"""
        downstream_layers = []
        
        if block_info.get('conv_position') == 'conv1':
            # conv1 -> conv2
            next_conv = layer_parts[:-1] + ['conv2']
            downstream_layers.append('net_feature_maps.' + '.'.join(next_conv))
            
        elif block_info.get('conv_position') == 'conv2':
            # conv2 -> conv3
            next_conv = layer_parts[:-1] + ['conv3']
            downstream_layers.append('net_feature_maps.' + '.'.join(next_conv))
            
        elif block_info.get('conv_position') == 'conv3':
            # conv3 -> 下一個 block 的 conv1 或下一個 layer group
            downstream_layers.extend(self._find_next_block_layers(layer_parts, block_info))
        
        return downstream_layers

    def _find_skip_connections_and_downsample(self, layer_parts, block_info):
        """
        找出跳躍連接相關層級和 downsample 層
        
        Returns:
            tuple: (skip_connections, downsample_layer)
        """
        skip_connections = []
        downsample_layer = None
        
        try:
            # ResNet 跳躍連接分析
            if block_info.get('is_first_block') and block_info.get('group') != 'layer1':
                # 第一個 block 通常有 downsample 層
                downsample_path = f"net_feature_maps.{block_info['group']}.0.downsample"
                if self._verify_layer_exists(downsample_path):
                    downsample_layer = downsample_path
                    skip_connections.append(downsample_path)
            
            # 檢查當前 block 是否有 downsample（針對 conv3 層）
            if block_info.get('conv_position') == 'conv3':
                # 構建當前 block 的 downsample 路徑
                current_block_path = f"net_feature_maps.{block_info['group']}.{block_info['block_index']}.downsample"
                
                if self._verify_layer_exists(current_block_path):
                    downsample_layer = current_block_path
                    skip_connections.append(current_block_path)
                
                # conv3 的輸出會與跳躍連接相加
                skip_connections.append('residual_addition')
            
            # 特殊情況：檢查是否為最後一個 block 的 conv3
            if (block_info.get('conv_position') == 'conv3' and 
                not self._has_next_block(block_info)):
                # 最後一個 block 的 conv3 可能影響整個 layer group 的輸出
                skip_connections.append('layer_group_output')
            
            return skip_connections, downsample_layer
            
        except Exception as e:
            print(f"[ERROR] 查找跳躍連接失敗: {e}")
            return [], None

    def _has_next_block(self, block_info):
        """檢查是否有下一個 block"""
        try:
            current_group = block_info['group']
            current_block = block_info['block_index']
            
            # 檢查同一 layer group 中的下一個 block
            next_block_path = f"net_feature_maps.{current_group}.{current_block + 1}"
            return self._verify_layer_exists(next_block_path)
            
        except:
            return False


    def _find_next_block_layers(self, layer_parts, block_info):
        """找出下一個 block 的相關層級"""
        next_layers = []
        
        current_group = block_info['group']
        current_block = block_info['block_index']
        
        # 嘗試找下一個 block
        next_block_layer = f"net_feature_maps.{current_group}.{current_block + 1}.conv1"
        if self._verify_layer_exists(next_block_layer):
            next_layers.append(next_block_layer)
        else:
            # 嘗試下一個 layer group
            next_group_mapping = {
                'layer1': 'layer2',
                'layer2': 'layer3', 
                'layer3': 'layer4'
            }
            
            if current_group in next_group_mapping:
                next_group = next_group_mapping[current_group]
                next_group_layer = f"net_feature_maps.{next_group}.0.conv1"
                if self._verify_layer_exists(next_group_layer):
                    next_layers.append(next_group_layer)
        
        return next_layers

    def _determine_pruning_strategy(self, layer_type, block_info):
        """確定剪枝策略"""
        
        if layer_type == 'standalone':
            return 'independent_pruning'
        
        elif layer_type == 'residual':
            if block_info.get('is_downsample_layer'):
                return 'coordinated_pruning_with_downsample'
            elif block_info.get('conv_position') == 'conv3':
                return 'residual_aware_pruning'
            else:
                return 'sequential_pruning'
        
        return 'standard_channel_pruning'

    def _verify_layer_exists(self, layer_name):
        """驗證層級是否存在於網路中"""
        try:
            layer = self.print_detailed_layers()
            layer_name = layer_name.replace('net_feature_maps.', '')
            return True if layer_name in layer else False
        except:
            return False
        
    def _needs_input_channel_fix(self, layer_parts, block_info):
        """判斷是否需要輸入通道修正"""
        try:
            # conv3 層通常需要輸入修正，因為它們影響殘差連接
            if block_info.get('conv_position') == 'conv3':
                return True
            
            # 第一個 block 的 conv1 可能需要修正
            if (block_info.get('is_first_block') and 
                block_info.get('conv_position') == 'conv1'):
                return True
            
            return False
            
        except:
            return False

    def print_detailed_layers(self, if_print=False):
        """列印詳細的層級資訊，包含參數數量"""
        print("=== net_feature_maps 詳細層級資訊 ===")
        get = []
        if hasattr(self.prune_network, 'net_feature_maps'):
            feature_maps = self.prune_network.net_feature_maps
            
            def print_detailed_info(module, prefix=''):
                for name, child in module.named_children():
                    full_name = f"{prefix}.{name}" if prefix else name
                    get.append( full_name )
                    # 獲取層級資訊
                    layer_info = f"{full_name}: {type(child).__name__}"
                    
                    # 添加參數資訊
                    if hasattr(child, 'weight') and child.weight is not None:
                        layer_info += f" - Weight: {child.weight.shape}"
                    
                    if hasattr(child, 'out_channels'):
                        layer_info += f" - Out Channels: {child.out_channels}"
                    if hasattr(child, 'num_features'):
                        layer_info += f" - Num Features: {child.num_features}"
                    if hasattr(child, 'in_channels'):
                        layer_info += f" - In Channels: {child.in_channels}"

                    if if_print:
                        print(layer_info)
                    
                    # 遞迴處理子模組
                    if len(list(child.children())) > 0:
                        print_detailed_info(child, full_name)
            
            print_detailed_info(feature_maps)
            return get
        else:
            print("找不到 net_feature_maps 模組")
