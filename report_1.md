#### 實驗結果
1. iter 1500 per layer + 0.3 剪枝率：效果不算太差 但 mAP 0.35 左右
    總參數量: 10,169,478
    可訓練參數量: 10,169,478
    模型存儲大小: 39.05 MB
    剪枝網路參數統計:

    總參數量: 7,836,568
    可訓練參數量: 7,836,568
    模型存儲大小: 30.14 MB
2. AI 設計的比例和次數
the_map_of_iter_and_prune_ratio = {
    # Layer 1 系列 - 前端特徵提取層
    "layer1.0.conv1": {"ratio": 0.95, "iter": 3000},
    "layer1.0.conv2": {"ratio": 0.95, "iter": 3000},
    "layer1.1.conv1": {"ratio": 0.72, "iter": 2513},
    "layer1.1.conv2": {"ratio": 0.64, "iter": 2326},
    "layer1.2.conv1": {"ratio": 0.31, "iter": 1630},
    "layer1.2.conv2": {"ratio": 0.39, "iter": 1807},
    
    # Layer 2 系列 - 中層特徵處理層
    "layer2.0.conv1": {"ratio": 0.36, "iter": 1725},
    "layer2.0.conv2": {"ratio": 0.31, "iter": 1629},
    "layer2.1.conv1": {"ratio": 0.25, "iter": 1500},
    "layer2.1.conv2": {"ratio": 0.25, "iter": 1500},
    "layer2.2.conv1": {"ratio": 0.36, "iter": 1744},
    "layer2.2.conv2": {"ratio": 0.41, "iter": 1838},
    "layer2.3.conv1": {"ratio": 0.43, "iter": 1885},
    "layer2.3.conv2": {"ratio": 0.44, "iter": 1911},
    
    # Layer 3 系列 - 深層特徵抽象層
    "layer3.1.conv2": {"ratio": 0.44, "iter": 1914},
    "layer3.2.conv2": {"ratio": 0.38, "iter": 1779},
    "layer3.3.conv2": {"ratio": 0.75, "iter": 2573},
    "layer3.4.conv2": {"ratio": 0.37, "iter": 1763},
    "layer3.5.conv2": {"ratio": 0.25, "iter": 1500}
}
