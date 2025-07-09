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
        keep_indices = self.get_prune_layers_keep_indices(layer_name)
        
        if not keep_indices:
            print(f"[WARNING] {layer_name} 無保留索引，跳過剪枝")
            return
        
        # 2. 解析層級依賴關係
        dependencies = self.resolve_layer_dependencies(layer_name)
        
        # 3. 剪枝當前層的輸出通道
        self._prune_out_channel(layer_name, keep_indices)
        
        # 4. 剪枝對應的 BatchNorm 層
        if dependencies.get('bn_layer'):
            self._prune_batchnorm_layer(dependencies['bn_layer'], keep_indices)
        
        # 5. 剪枝下游層的輸入通道
        if dependencies.get('next_layers'):
            for next_layer in dependencies['next_layers']:
                self._prune_next_layer_inputs(layer_name, next_layer, keep_indices)
        
        # 6. 處理殘差連接
        if dependencies.get('downsample_layer'):
            self._prune_downsample_connection(dependencies['downsample_layer'], keep_indices)
        
        # 7. 處理輸入通道修正（針對特殊連接）
        if dependencies.get('needs_input_fix'):
            self._prune_in_channel_for_connection_fix(layer_name, keep_indices)
        
        # 8. 更新通道索引追蹤
        self.track_channel_indices(layer_name, keep_indices)
    
        print(f"[PRUNE] 完成剪枝層級: {layer_name}")



    def _prune_next_layer_inputs(self, current_layer_name, next_layer_name, keep_indices):
        pass

    def _prune_in_channel_for_connection_fix(self, layer_name, keep_indexes):
        pass

    def _prune_batchnorm_layer(self, bn_layer, keep_indices):
        pass

    def _prune_downsample_connection(self, downsample_layer, keep_indices):
        pass

    def track_channel_indices(self, layer_name, keep_indices):
        pass

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
            'skip_connections': [],
            'dependency_type': 'conv',
            'pruning_strategy': 'channel_pruning'
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
            
            # 5. 處理 ResNet 跳躍連接
            if layer_type == 'residual':
                skip_connections = self._find_skip_connections(layer_parts, block_info)
                dependencies['skip_connections'] = skip_connections
            
            # 6. 確定剪枝策略
            pruning_strategy = self._determine_pruning_strategy(layer_type, block_info)
            dependencies['pruning_strategy'] = pruning_strategy
            
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

    def _find_skip_connections(self, layer_parts, block_info):
        """找出跳躍連接相關層級"""
        skip_connections = []
        
        # ResNet 跳躍連接分析
        if block_info.get('is_first_block') and block_info.get('group') != 'layer1':
            # 第一個 block 通常有 downsample 層
            downsample_layer = f"net_feature_maps.{block_info['group']}.0.downsample.0"
            skip_connections.append(downsample_layer)
        
        # 檢查是否影響殘差連接
        if block_info.get('conv_position') == 'conv3':
            # conv3 的輸出會與跳躍連接相加
            skip_connections.append('residual_addition')
        
        return skip_connections

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
                    elif hasattr(child, 'num_features'):
                        layer_info += f" - Num Features: {child.num_features}"
                    
                    if if_print:
                        print(layer_info)
                    
                    # 遞迴處理子模組
                    if len(list(child.children())) > 0:
                        print_detailed_info(child, full_name)
            
            print_detailed_info(feature_maps)
            return get
        else:
            print("找不到 net_feature_maps 模組")
