import os
import csv
from typing import Dict, List, Optional

class PruneDBControler:
    def __init__(self, path):
        self._path = path

    def initial(self):
        """初始化 CSV 檔案 - 強制刪除重新創建"""
        f = self._path
        
        # 檢查檔案是否存在，如果存在則刪除
        if os.path.exists(f):
            print(f"檔案 {f} 已存在，將刪除並重新創建。")
            os.remove(f)
        
        # 創建新檔案
        print(f"創建新檔案: {f}")
        with open(f, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["layer", "original_channel_num", "num_of_keep_channel", "keep_index"])
        
        print(f"檔案 {f} 初始化完成。")

    def write_data(self, layer, original_channel_num, num_of_keep_channel, keep_index):
        """寫入剪枝數據到 CSV 檔案，如果 layer 已存在則覆蓋"""
        try:
            keep_index = keep_index.tolist()
        except:
            pass
        print(keep_index)
        print(type(keep_index))
        print(str(keep_index))
        
        # 檢查檔案是否存在
        if not os.path.exists(self._path):
            # 檔案不存在，創建新檔案
            with open(self._path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["layer", "original_channel_num", "num_of_keep_channel", "keep_index"])
        
        # 讀取現有數據
        existing_data = []
        layer_found = False
        
        try:
            with open(self._path, 'r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                header = next(reader)  # 讀取標題行
                existing_data.append(header)
                
                for row in reader:
                    if row[0] == layer:  # 找到相同的 layer
                        # 覆蓋該行數據
                        keep_index_str = str(keep_index) if isinstance(keep_index, list) else str([keep_index])
                        new_row = [layer, original_channel_num, num_of_keep_channel, keep_index_str]
                        existing_data.append(new_row)
                        layer_found = True
                        print(f"[INFO] 覆蓋現有層級數據: {layer}")
                    else:
                        # 保留其他行
                        existing_data.append(row)
        
        except FileNotFoundError:
            # 檔案不存在的情況已在上面處理
            pass
        
        # 如果沒有找到相同的 layer，添加新記錄
        if not layer_found:
            keep_index_str = str(keep_index) if isinstance(keep_index, list) else str([keep_index])
            new_row = [layer, original_channel_num, num_of_keep_channel, keep_index_str]
            existing_data.append(new_row)
            print(f"[INFO] 添加新層級數據: {layer}")
        
        # 重寫整個檔案
        with open(self._path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(existing_data)

    def _get_all_data_by_layer(self, layer) -> Dict:
        """
        根據層名稱獲取所有相關數據
        """
        if not os.path.exists(self._path):
            return {}
        
        try:
            with open(self._path, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    if row['layer'] == layer:
                        # 解析 keep_index 字串為整數列表
                        keep_index_str = row['keep_index']
                        if keep_index_str:
                            # 使用 ast.literal_eval 安全解析 list 字串
                            import ast
                            try:
                                keep_index = ast.literal_eval(keep_index_str)
                                # 確保結果是 list
                                if not isinstance(keep_index, list):
                                    keep_index = [keep_index]
                            except (ValueError, SyntaxError):
                                # 降級處理：嘗試舊格式解析
                                keep_index = [int(x.strip()) for x in keep_index_str.split(',') if x.strip()]
                        else:
                            keep_index = []
                        
                        return {
                            'layer': row['layer'],
                            'original_channel_num': int(row['original_channel_num']),
                            'num_of_keep_channel': int(row['num_of_keep_channel']),
                            'keep_index': keep_index
                        }
                
                return {}
                
        except (FileNotFoundError, ValueError, KeyError) as e:
            print(f"讀取數據時發生錯誤: {e}")
            return {}


    def get_layer_keep_indices_num(self, layer) -> int:
        """
        獲取指定層保留的通道數量
        
        Args:
            layer: 層名稱
            
        Returns:
            int: 保留的通道數量，如果找不到則返回 0
        """
        data = self._get_all_data_by_layer(layer)
        if data:
            return data['num_of_keep_channel']
        return 0

    def get_layer_keep_indices(self, layer) -> List[int]:
        """
        獲取指定層保留的通道索引列表
        
        Args:
            layer: 層名稱
            
        Returns:
            List[int]: 保留的通道索引列表，如果找不到則返回空列表
        """
        data = self._get_all_data_by_layer(layer)
        if data:
            return data['keep_index']
        return []

    def get_layer_original_channel_num(self, layer) -> int:
        """
        獲取指定層原始的通道數量
        
        Args:
            layer: 層名稱
            
        Returns:
            int: 原始通道數量，如果找不到則返回 0
        """
        data = self._get_all_data_by_layer(layer)
        if data:
            return data['original_channel_num']
        return 0

    # 額外的實用方法
    def layer_exists(self, layer) -> bool:
        """檢查指定層是否存在於數據庫中"""
        data = self._get_all_data_by_layer(layer)
        return bool(data)

    def get_all_layers(self) -> List[str]:
        """獲取所有已記錄的層名稱"""
        if not os.path.exists(self._path):
            return []
        
        layers = []
        try:
            with open(self._path, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row['layer'] not in layers:
                        layers.append(row['layer'])
        except Exception as e:
            print(f"讀取層名稱時發生錯誤: {e}")
        
        return layers

    def get_pruning_summary(self) -> Dict:
        """獲取剪枝摘要統計"""
        if not os.path.exists(self._path):
            return {}
        
        summary = {
            'total_layers': 0,
            'total_original_channels': 0,
            'total_kept_channels': 0,
            'average_pruning_ratio': 0.0,
            'layers_info': []
        }
        
        try:
            with open(self._path, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    original_channels = int(row['original_channel_num'])
                    kept_channels = int(row['num_of_keep_channel'])
                    pruning_ratio = (original_channels - kept_channels) / original_channels * 100
                    
                    summary['total_layers'] += 1
                    summary['total_original_channels'] += original_channels
                    summary['total_kept_channels'] += kept_channels
                    
                    summary['layers_info'].append({
                        'layer': row['layer'],
                        'original_channels': original_channels,
                        'kept_channels': kept_channels,
                        'pruning_ratio': pruning_ratio
                    })
                
                if summary['total_original_channels'] > 0:
                    summary['average_pruning_ratio'] = (
                        (summary['total_original_channels'] - summary['total_kept_channels']) / 
                        summary['total_original_channels'] * 100
                    )
                    
        except Exception as e:
            print(f"計算摘要時發生錯誤: {e}")
        
        return summary

    def update_layer_data(self, layer, original_channel_num, num_of_keep_channel, keep_index):
        """更新指定層的數據（如果存在則更新，否則新增）"""
        if not os.path.exists(self._path):
            self.initial()
        
        # 讀取所有數據
        all_data = []
        layer_found = False
        
        try:
            with open(self._path, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row['layer'] == layer:
                        # 更新現有數據
                        if isinstance(keep_index, list):
                            keep_index_str = ','.join(map(str, keep_index))
                        else:
                            keep_index_str = str(keep_index)
                        
                        row['original_channel_num'] = str(original_channel_num)
                        row['num_of_keep_channel'] = str(num_of_keep_channel)
                        row['keep_index'] = keep_index_str
                        layer_found = True
                    
                    all_data.append(row)
            
            # 如果層不存在，添加新數據
            if not layer_found:
                if isinstance(keep_index, list):
                    keep_index_str = ','.join(map(str, keep_index))
                else:
                    keep_index_str = str(keep_index)
                
                all_data.append({
                    'layer': layer,
                    'original_channel_num': str(original_channel_num),
                    'num_of_keep_channel': str(num_of_keep_channel),
                    'keep_index': keep_index_str
                })
            
            # 重寫檔案
            with open(self._path, 'w', newline='', encoding='utf-8') as file:
                if all_data:
                    fieldnames = ['layer', 'original_channel_num', 'num_of_keep_channel', 'keep_index']
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_data)
                    
        except Exception as e:
            print(f"更新數據時發生錯誤: {e}")
