import csv
import numpy as np
import torch
import hashlib
import os
from datetime import datetime

class GradientCSVStorage:
    def __init__(self, csv_path="lcp_gradients.csv"):
        self.csv_path = csv_path
        self.init_csv()
    
    def init_csv(self):
        """初始化 CSV 檔案，如果不存在則創建"""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    'layer_name', 'config_hash', 'use_image_num', 'random_seed', 
                    'lambda_rate', 'gradient_file', 'importance_scores', 
                    'gradient_shape', 'timestamp'
                ])
    
    def _generate_config_hash(self, use_image_num, random_seed, lambda_rate=1.0):
        """生成配置的唯一雜湊值"""
        config_str = f"{use_image_num}_{random_seed}_{lambda_rate}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def save_gradient(self, layer_name, gradient_tensor, use_image_num, random_seed, lambda_rate=1.0):
        """存儲梯度到 CSV 和對應的 numpy 檔案"""
        config_hash = self._generate_config_hash(use_image_num, random_seed, lambda_rate)
        
        # 生成梯度檔案名
        gradient_filename = f"grad_{layer_name.replace('.', '_')}_{config_hash}.npy"
        gradient_filepath = os.path.join(os.path.dirname(self.csv_path), gradient_filename)
        
        # 序列化梯度並保存為 .npy 檔案
        grad_numpy = gradient_tensor.detach().cpu().numpy()
        np.save(gradient_filepath, grad_numpy)
        
        # 計算 channel importance
        importance = np.sum(grad_numpy**2, axis=(1, 2, 3))
        importance_str = ','.join(map(str, importance))
        
        # 檢查是否已存在相同配置
        if self._record_exists(layer_name, config_hash):
            print(f"[LOG] 更新現有記錄: {layer_name}, 配置: {config_hash}")
            self._update_record(layer_name, config_hash, gradient_filename, importance_str, grad_numpy.shape)
        else:
            print(f"[LOG] 新增記錄: {layer_name}, 配置: {config_hash}")
            self._add_record(layer_name, config_hash, use_image_num, random_seed, 
                           lambda_rate, gradient_filename, importance_str, grad_numpy.shape)
    
    def _record_exists(self, layer_name, config_hash):
        """檢查記錄是否已存在"""
        try:
            with open(self.csv_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row['layer_name'] == layer_name and row['config_hash'] == config_hash:
                        return True
        except FileNotFoundError:
            return False
        return False
    
    def _add_record(self, layer_name, config_hash, use_image_num, random_seed, 
                   lambda_rate, gradient_filename, importance_str, shape):
        """新增記錄到 CSV"""
        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                layer_name, config_hash, use_image_num, random_seed,
                lambda_rate, gradient_filename, importance_str,
                str(shape), datetime.now().isoformat()
            ])
    
    def _update_record(self, layer_name, config_hash, gradient_filename, importance_str, shape):
        """更新現有記錄"""
        rows = []
        with open(self.csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames
            
            for row in reader:
                if row['layer_name'] == layer_name and row['config_hash'] == config_hash:
                    row['gradient_file'] = gradient_filename
                    row['importance_scores'] = importance_str
                    row['gradient_shape'] = str(shape)
                    row['timestamp'] = datetime.now().isoformat()
                rows.append(row)
        
        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    
    def load_gradient(self, layer_name, use_image_num, random_seed, lambda_rate=1.0):
        """從 CSV 和 numpy 檔案讀取梯度"""
        config_hash = self._generate_config_hash(use_image_num, random_seed, lambda_rate)
        
        try:
            with open(self.csv_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row['layer_name'] == layer_name and row['config_hash'] == config_hash:
                        gradient_file = row['gradient_file']
                        gradient_filepath = os.path.join(os.path.dirname(self.csv_path), gradient_file)
                        
                        if os.path.exists(gradient_filepath):
                            grad_numpy = np.load(gradient_filepath)
                            print(f"[LOG] 梯度已讀取: {layer_name}, 檔案: {gradient_file}")
                            return grad_numpy
                        else:
                            print(f"[WARNING] 梯度檔案不存在: {gradient_filepath}")
                            return None
        except FileNotFoundError:
            print(f"[WARNING] CSV 檔案不存在: {self.csv_path}")
            return None
        
        return None
    
    def load_importance(self, layer_name, use_image_num, random_seed, lambda_rate=1.0):
        """直接從 CSV 讀取 channel importance（無需載入完整梯度）"""
        config_hash = self._generate_config_hash(use_image_num, random_seed, lambda_rate)
        
        try:
            with open(self.csv_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row['layer_name'] == layer_name and row['config_hash'] == config_hash:
                        importance_str = row['importance_scores']
                        importance = np.array([float(x) for x in importance_str.split(',')])
                        print(f"[LOG] Channel importance 已讀取: {layer_name}")
                        return importance
        except FileNotFoundError:
            return None
        
        return None
