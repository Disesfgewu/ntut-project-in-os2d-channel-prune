import torch

def count_parameters(model):
    """
    計算模型的參數數量
    Returns:
      total_params: 包含所有參數
      trainable_params: 只包含 requires_grad=True 的參數
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

import io

def estimate_model_size(model):
    """
    將模型序列化到緩衝區，估算存檔大小（MB）
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mb = buffer.getbuffer().nbytes / (1024 ** 2)
    return size_mb

def show_network_status( orig_net, prune_net ):
    print( "原始網路參數統計:\n" )
    model = orig_net
    total, trainable = count_parameters(model)
    print(f"總參數量: {total:,}")
    print(f"可訓練參數量: {trainable:,}")

    size_mb = estimate_model_size(model)
    print(f"模型存儲大小: {size_mb:.2f} MB")

    print( "剪枝網路參數統計:\n" )
    model = prune_net
    total, trainable = count_parameters(model)
    print(f"總參數量: {total:,}")
    print(f"可訓練參數量: {trainable:,}")

    size_mb = estimate_model_size(model)
    print(f"模型存儲大小: {size_mb:.2f} MB")

    