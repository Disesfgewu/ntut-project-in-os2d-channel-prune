import torch
import os

class LCPReconstruction:
    def __init__(self, prune_db, pruner, prune_net=None):
        self._prune_db = prune_db
        self._pruner = pruner
        self._prune_net = prune_net
        self.map = {}
        self._pruned_layers = []
        self._all_layers = [
            'conv1',
            'layer1.0.conv1',
            'layer1.0.conv2',
            'layer1.1.conv1',
            'layer1.1.conv2',
            'layer1.2.conv1',
            'layer1.2.conv2',
            'layer2.0.conv1',
            'layer2.0.conv2',
            'layer2.1.conv1',
            'layer2.1.conv2',
            'layer2.2.conv1',
            'layer2.2.conv2',
            'layer2.3.conv1',
            'layer2.3.conv2',
            'layer3.0.conv1',
            'layer3.0.conv2',
            'layer3.1.conv1',
            'layer3.1.conv2',
            'layer3.2.conv1',
            'layer3.2.conv2',
            'layer3.3.conv1',
            'layer3.3.conv2',
            'layer3.4.conv1',
            'layer3.4.conv2',
            'layer3.5.conv1',
            'layer3.5.conv2',
        ]

        self._init_for_map()

    def _init_for_map(self):
        for layer in self._all_layers:
            self.map[layer] = {
                'in': [],
                'out': [],
                'original_in_channel_num': 0,
                'original_out_channel_num': 0,
                'keep_index': [],
                'bn_features': [],
                'original_bn_features': 0
            }

    def get_output_channels(self, conv_weight_tensor):
        """
        獲取卷積層的輸出通道資訊 - 返回 List 格式
        
        Args:
            conv_weight_tensor: 卷積層權重張量 [out_channels, in_channels, H, W]
        
        Returns:
            dict: 包含輸出通道資訊的字典，其中 output_channels_list 為 Python List
        """
        import torch
        
        if len(conv_weight_tensor.shape) != 4:
            raise ValueError("輸入張量必須是4維卷積權重格式 [out_channels, in_channels, H, W]")
        
        out_channels_count = conv_weight_tensor.shape[0]
        in_channels_count = conv_weight_tensor.shape[1]
        kernel_size = (conv_weight_tensor.shape[2], conv_weight_tensor.shape[3])
        
        # 將所有輸出通道轉換為 Python List
        output_channels_list = []
        for i in range(out_channels_count):
            channel_tensor = conv_weight_tensor[i]  # [in_channels, H, W]
            # 轉換為 Python List (3D list)
            channel_list = channel_tensor.tolist()
            output_channels_list.append(channel_list)
        
        return {
            'out_channels_count': out_channels_count,
            'in_channels_count': in_channels_count,
            'kernel_size': kernel_size,
            'output_channels_list': output_channels_list,  # 新增：List 格式的輸出通道
            'weight_tensor': conv_weight_tensor,
            'total_params': conv_weight_tensor.numel()
        }

    def get_specific_output_channel(self, conv_weight_tensor, channel_index):
        """
        獲取特定輸出通道的權重
        
        Args:
            conv_weight_tensor: 卷積層權重張量 [out_channels, in_channels, H, W]
            channel_index: 要獲取的輸出通道索引
        
        Returns:
            torch.Tensor: 特定輸出通道的權重 [in_channels, H, W]
        """
        
        if len(conv_weight_tensor.shape) != 4:
            raise ValueError("輸入張量必須是4維卷積權重格式")
        
        out_channels_count = conv_weight_tensor.shape[0]
        
        if channel_index >= out_channels_count or channel_index < 0:
            raise ValueError(f"通道索引 {channel_index} 超出範圍 [0, {out_channels_count-1}]")
        
        # 獲取特定輸出通道的權重
        channel_weight = conv_weight_tensor[channel_index]  # [in_channels, H, W]
        
        # 計算統計資訊
        stats = {
            'channel_index': channel_index,
            'weight': channel_weight,
            'shape': channel_weight.shape,
            'mean': channel_weight.mean().item(),
            'std': channel_weight.std().item(),
            'min': channel_weight.min().item(),
            'max': channel_weight.max().item(),
            'norm': torch.norm(channel_weight).item()
        }
        
        return stats

    def update_output_channel(self, conv_weight_tensor, keep_indices, orig_num):
        """
        根據 keep_indices 恢復輸出通道到原始數量
        
        Args:
            conv_weight_tensor: 剪枝後的卷積權重 [pruned_channels, in_channels, H, W]
            keep_indices: 保留的通道索引列表
            orig_num: 原始通道數量
        
        Returns:
            torch.Tensor: 恢復後的卷積權重 [orig_num, in_channels, H, W]
        """
        if len(conv_weight_tensor.shape) != 4:
            raise ValueError("輸入張量必須是4維卷積權重格式")
        
        # 獲取設備和數據類型
        device = conv_weight_tensor.device
        dtype = conv_weight_tensor.dtype
        
        # 創建新的全零張量
        new_shape = (orig_num, *conv_weight_tensor.shape[1:])
        new_tensor = torch.zeros(new_shape, dtype=dtype, device=device)
        
        # 將剪枝後的權重映射回原始位置
        for j, keep_idx in enumerate(keep_indices):
            if j < conv_weight_tensor.shape[0] and keep_idx < orig_num:
                new_tensor[keep_idx] = conv_weight_tensor[j]
        
        # 直接更新原始張量的數據
        conv_weight_tensor.data = new_tensor
        
        return conv_weight_tensor

    def update_input_channel(self, conv_weight_tensor, keep_indices, orig_num):
        """
        根據 keep_indices 恢復輸入通道到原始數量
        
        Args:
            conv_weight_tensor: 剪枝後的卷積權重 [out_channels, pruned_in_channels, H, W]
            keep_indices: 保留的輸入通道索引列表
            orig_num: 原始輸入通道數量
        
        Returns:
            torch.Tensor: 恢復後的卷積權重 [out_channels, orig_num, H, W]
        """
        if len(conv_weight_tensor.shape) != 4:
            raise ValueError("輸入張量必須是4維卷積權重格式")
        
        # 獲取設備和數據類型
        device = conv_weight_tensor.device
        dtype = conv_weight_tensor.dtype
        
        # 創建新的全零張量 - 注意這裡是恢復第1維(輸入通道)
        new_shape = (conv_weight_tensor.shape[0], orig_num, *conv_weight_tensor.shape[2:])
        new_tensor = torch.zeros(new_shape, dtype=dtype, device=device)
        
        # 將剪枝後的權重映射回原始位置
        for j, keep_idx in enumerate(keep_indices):
            if j < conv_weight_tensor.shape[1] and keep_idx < orig_num:
                # 注意這裡是操作第1維(輸入通道維度)
                new_tensor[:, keep_idx] = conv_weight_tensor[:, j]
        
        # 直接更新原始張量的數據
        conv_weight_tensor.data = new_tensor
        
        return conv_weight_tensor

    def update_bn_features(self, bn_param_tensor, keep_indices, orig_num):
        """
        統一處理所有 BatchNorm 參數恢復
        
        Args:
            bn_param_tensor: 剪枝後的 BatchNorm 參數 [pruned_features]
            keep_indices: 保留的特徵索引列表
            orig_num: 原始特徵數量
        
        Returns:
            torch.Tensor: 恢復後的 BatchNorm 參數 [orig_num]
        """
        print( bn_param_tensor )
        if len(bn_param_tensor.shape) != 1:
            return None
        
        # 獲取設備和數據類型
        device = bn_param_tensor.device
        dtype = bn_param_tensor.dtype
        
        # 創建新的全零張量
        new_tensor = torch.zeros(orig_num, dtype=dtype, device=device)
        
        # 將剪枝後的參數映射回原始位置
        for j, keep_idx in enumerate(keep_indices):
            if j < bn_param_tensor.shape[0] and keep_idx < orig_num:
                new_tensor[keep_idx] = bn_param_tensor[j]
        
        # 直接更新原始張量的數據
        bn_param_tensor.data = new_tensor
        
        return bn_param_tensor

    def get_dependency(self, layer_name):
        layer_name = f"net_feature_maps.{layer_name}" if not layer_name.startswith("net_feature_maps") else layer_name
        return self._pruner.resolve_layer_dependencies(layer_name)

    def _get_channel_list(self, layer_name):
        layer_name = f"net_feature_maps.{layer_name}" if not layer_name.startswith("net_feature_maps") else layer_name
        return self._prune_db._get_all_data_by_layer(layer_name)['keep_index']

    def _get_channel_num_orig(self, layer_name):
        layer_name = f"net_feature_maps.{layer_name}" if not layer_name.startswith("net_feature_maps") else layer_name
        return self._prune_db._get_all_data_by_layer(layer_name)['original_channel_num']

    def _reshape_layer_name(self, layer_name):
        partition = layer_name.split('.')
        if len(partition) > 2:
            return f"{partition[-3]}.{partition[-2]}.{partition[-1]}"
        else:
            return partition[-1]

    def _extract_layer_name(self, layer_name):
        filter = [
            '.weight',
            '.bias',
            '.running_mean',
            '.running_var',
            '.num_batches_tracked',
        ]
        for f in filter:
            if layer_name.endswith(f):
                return layer_name[:-len(f)]
        return layer_name

    def _extract_all_channel_information_for_layers(self, layers):
        for layer in layers:
            dependency = self.get_dependency(f'net_feature_maps.{layer}')
            if dependency['target'] != None:
                dependency['target'] = dependency['target'].replace('net_feature_maps.', '')
                self.map[dependency['target']]['out'] = self._get_channel_list(layer)
                self.map[dependency['target']]['original_out_channel_num'] = self._get_channel_num_orig(layer)
            if dependency['batch_norm'] != None:
                self.map[dependency['target']]['bn_features'] = self._get_channel_list(layer)
                self.map[dependency['target']]['original_bn_features'] = self._get_channel_num_orig(layer)
            if dependency['downstream_layers'] != None:
                for next in dependency['downstream_layers']:
                    next = next.replace('net_feature_maps.', '')
                    self.map[next]['in'] = self._get_channel_list(layer)
                    self.map[next]['original_in_channel_num'] = self._get_channel_num_orig(layer)

    def load_checkpoint_with_pruned_net(self, checkpoint_file_path, optimizer=None, model_name=None):
        if not os.path.exists(checkpoint_file_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file_path}")
        
        print(f"Loading checkpoint from {checkpoint_file_path}")
        
        # 載入 checkpoint
        checkpoint = torch.load(checkpoint_file_path, map_location='cpu')
        
        # 檢查是否有剪枝記錄
        if hasattr(self, '_prune_db') and self._prune_db is not None:
            print("Restoring pruned parameters to original dimensions...")
            
            # 恢復剪枝參數到原始維度
            restored_state_dict = self._restore_pruned_parameters(
                checkpoint.get("net", {}), 
                self._prune_db
            )
            print( restored_state_dict )
            # 載入恢復後的網路狀態
            if restored_state_dict:
                self._prune_net.load_state_dict(restored_state_dict)
                print("Successfully loaded restored pruned network state")
            else:
                print("No network state found in checkpoint")
        else:
            # 沒有剪枝記錄，直接載入（可能會失敗）
            print("No prune database found, attempting direct load...")
            if "net" in checkpoint:
                self._prune_net.load_state_dict(checkpoint["net"])
                print("Successfully loaded pruned network state")
            else:
                print("No network state found in checkpoint")
        
        # 載入 optimizer 狀態（如果提供）
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("Successfully loaded optimizer state")
        elif optimizer is not None:
            print("Optimizer provided but no optimizer state found in checkpoint")
        
        # 檢查是否需要移到 GPU
        if next(self._prune_net.parameters()).device.type == 'cuda':
            self._prune_net = self._prune_net.cuda()
            print("Moved pruned network to CUDA")
        
        # 返回 checkpoint 以便存取額外欄位
        model_info = f" for model '{model_name}'" if model_name else ""
        print(f"Successfully loaded checkpoint{model_info}")
        
        return checkpoint

    def _restore_pruned_parameters(self, pruned_state_dict, prune_db):
        """
        恢復剪枝參數到原始維度
        Args:
            pruned_state_dict: 剪枝後的模型狀態字典
            prune_db: 剪枝數據庫控制器
        Returns:
            restored_state_dict: 恢復後的模型狀態字典
        """
        restored_state_dict = {}
        all_layers = prune_db.get_all_layers()
    
        pruned_layers = [layer for layer in all_layers if layer.startswith('layer')]
        # print( all_layers )
        # print( pruned_layers )
        print(f"Found {len(all_layers)} total layers in database")
        print(f"Filtered to {len(pruned_layers)} layers starting with 'layer'")
        self._extract_all_channel_information_for_layers(pruned_layers)
        
        for layer_name, params in pruned_state_dict.items():
            layer = self._extract_layer_name(layer_name).replace('net_feature_maps.', '')
            layer = self._reshape_layer_name(layer)
            try:
                if layer.endswith('conv1') or layer.endswith('conv2'):
                    if self.map[layer]['out'] != []:
                        self._pruned_layers.append(layer)
                        out_channels_update = self.update_output_channel(params, self.map[layer]['out'], self.map[layer]['original_out_channel_num'])
                        restored_state_dict[layer_name] = out_channels_update
                    elif self.map[layer]['in'] != []:
                        in_channels_update = self.update_input_channel(params, self.map[layer]['in'], self.map[layer]['original_in_channel_num'])
                        restored_state_dict[layer_name] = in_channels_update
                    else:
                        print(f"Skipping layer '{layer_name}' as it is not pruned")
                        restored_state_dict[layer_name] = params
                elif layer.endswith('bn1') or layer.endswith('bn2'):
                    get_layer = layer.replace('bn', 'conv')
                    if self.map[get_layer]['bn_features'] != []:
                        # 統一處理，不需要區分參數類型
                        bn_features_update = self.update_bn_features(
                            params,
                            self.map[get_layer]['bn_features'],
                            self.map[get_layer]['original_bn_features']
                        )
                        if bn_features_update is None:
                            print(f"Skipping layer '{layer_name}' as it has no valid BatchNorm parameters")
                            continue
                        restored_state_dict[layer_name] = bn_features_update
                        print(f"✅ 恢復 BatchNorm 參數: {layer_name}")
                    else:
                        print(f"Skipping layer '{layer_name}' as it is not pruned")
                        restored_state_dict[layer_name] = params
                else:
                    print(f"Skipping layer '{layer_name}' as it is not pruned")
                    restored_state_dict[layer_name] = params
            except Exception as e:
                print(f"Skipping layer '{layer_name}' as it is not pruned")
                restored_state_dict[layer_name] = params
            
        return restored_state_dict