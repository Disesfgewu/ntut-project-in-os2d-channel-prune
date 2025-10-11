import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
import torchvision.transforms as transforms
from os2d.utils import get_image_size_after_resize_preserving_aspect_ratio

class GenerativeAIDataset(Dataset):
    """
    一個客製化的 PyTorch Dataset，用於讀取由 gen_db.csv 定義的生成式 AI 探測影像。
    """
    def __init__(self, csv_path, img_normalization, resize_target=1500):
        """
        初始化 Dataset。

        Args:
            csv_path (str): gen_db.csv 檔案的路徑。
            img_normalization (dict): 包含 'mean' 和 'std' 的圖像標準化參數。
            resize_target (int): 影像 resize 的目標尺寸。
        """
        try:
            self.db = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"錯誤：找不到 gen_db.csv 於 '{csv_path}'。請先執行 create_gen_db.py。")

        self.resize_target = resize_target
        
        # 確保圖像轉換流程與您專案的其他部分完全一致
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=img_normalization["mean"], std=img_normalization["std"])
        ])
        
        print(f"✅ GenerativeAIDataset 初始化完成，共載入 {len(self.db)} 筆影像資料。")

    def __len__(self):
        """返回資料集的總大小。"""
        return len(self.db)

    def __getitem__(self, idx):
        """
        根據索引 idx 獲取一筆資料。

        Args:
            idx (int): 資料的索引。

        Returns:
            torch.Tensor: 經過預處理的圖像張量。
        """
        # 從 DataFrame 獲取影像路徑
        img_path = self.db.loc[idx, 'unique_path']

        try:
            # 載入影像
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"警告：找不到影像檔案 '{img_path}'，將返回一個空張量。")
            # 返回一個符合預期維度的空張量以避免訓練中斷
            return torch.zeros((3, self.resize_target, self.resize_target))

        # --- 圖像預處理 (與您 LCP 程式碼中的邏輯保持一致) ---
        # 1. 計算保持長寬比的 resize 尺寸
        h, w = get_image_size_after_resize_preserving_aspect_ratio(
            h=image.size[1],
            w=image.size[0],
            target_size=self.resize_target
        )
        
        # 2. resize 圖像
        image = image.resize((w, h))

        # 3. 應用轉換
        image_tensor = self.transform(image)

        return image_tensor

def build_gen_dataloader(cfg, img_normalization, csv_path='./src/db/gen_db.csv'):
    """
    一個輔助函式，用於創建 GenerativeAIDataset 並將其封裝到 PyTorch DataLoader 中。

    Args:
        cfg: OS2D 的配置物件。
        img_normalization (dict): 圖像標準化參數。
        csv_path (str): gen_db.csv 的路徑。

    Returns:
        torch.utils.data.DataLoader: 配置完成的 DataLoader 物件。
    """
    dataset = GenerativeAIDataset(
        csv_path=csv_path,
        img_normalization=img_normalization,
        # <--- 修正點：將 cfg.input.image_size 改為固定的 1500 --->
        resize_target=1500
    )

    gen_dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,  # 在訓練時通常會打亂數據
        num_workers=0,
        pin_memory=True # 如果使用 GPU，可以加速數據轉移
    )
    
    print("✅ 生成式 AI DataLoader 建立完成。")
    return gen_dataloader