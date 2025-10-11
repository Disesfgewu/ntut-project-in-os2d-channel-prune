import torch

print("Testing basic CUDA functionality...")
try:
    # 測試基本 CUDA 操作
    if torch.cuda.is_available():
        x = torch.randn(10, 10).cuda()
        y = torch.randn(10, 10).cuda()
        z = x + y
        print("基本 CUDA 操作成功")
        
        # 測試更複雜的操作
        z = torch.mm(x, y)
        print("矩陣運算成功")
        
    else:
        print("CUDA 不可用")
        
except Exception as e:
    print(f"GPU 測試失敗: {e}")
