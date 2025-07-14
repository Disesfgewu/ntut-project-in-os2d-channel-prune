import logging
import torch
from os2d.engine.train import trainval_loop

class LCPFineTune:
    def __init__(self, prune_net, dataloader_train, img_normalization, box_coder, cfg, optimizer, parameters):
        """
        LCP Fine-tune 類別 - 簡化版本
        
        Args:
            prune_net: 剪枝後的網路模型
            dataloader_train: 訓練數據加載器
            img_normalization: 圖像標準化參數
            box_coder: 邊界框編碼器
            cfg: 配置對象
            optimizer: 優化器
            parameters: 可訓練參數
        """
        self._prune_net = prune_net
        self.dataloader_train = dataloader_train
        self.img_normalization = img_normalization
        self.box_coder = box_coder
        self.cfg = cfg
        self.optimizer = optimizer
        self.parameters = parameters
        
        # 基礎設置
        self._logger = logging.getLogger("LCP.FineTune")
        self._device = 'cuda' if cfg.is_cuda else 'cpu'
        self._setup_logging()
        self._ensure_model_on_device()
    
    def _setup_logging(self):
        """設置日誌記錄"""
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def _ensure_model_on_device(self):
        """確保模型在正確設備上"""
        if self.cfg.is_cuda and next(self._prune_net.parameters()).device.type != 'cuda':
            self._prune_net = self._prune_net.cuda()

    def criterion(self, aux_net):
        pass

    def start_finetune(self, criterion, finetune_iterations=1000, reconstruction_weight=0.1, 
                      dataloaders_eval=None):
        """
        啟動 fine-tune 流程
        
        Args:
            criterion: 損失函數
            finetune_iterations: fine-tune 迭代次數
            reconstruction_weight: 重建損失權重 (預留參數)
            dataloaders_eval: 評估數據加載器列表
        """
        
        
        # 直接呼叫 trainval_loop
        trainval_loop(
            dataloader_train=self.dataloader_train,
            net=self._prune_net,
            cfg=self.cfg,
            criterion=criterion,
            optimizer=self.optimizer,
            dataloaders_eval=dataloaders_eval or []
        )
        
        self._logger.info("LCP Fine-tune 完成")
