from dataclasses import dataclass
import torch
import os

@dataclass
class TransformerConfig:
    # 模型参数 - 确保与transformer.py默认值一致
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    max_seq_length: int = 128
    
    # 训练参数
    batch_size: int = 32
    optimizer: str = 'Adam'
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 4000
    epochs: int = 20
    accumulation_steps: int = 8
    
    # 验证参数
    val_freq: int = 200
    early_stopping_patience: int = 5
    
    # 其他设置
    device: str = None
    checkpoint_dir: str = 'checkpoints'
    results_dir: str = 'results'
    seed: int = 42

    def __post_init__(self):
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 验证配置的一致性
        assert self.d_model % self.num_heads == 0, \
            f"d_model ({self.d_model}) 必须能被 num_heads ({self.num_heads}) 整除"
        
        if self.device == 'cuda':
            try:
                torch.cuda.empty_cache()
                test_tensor = torch.zeros(1, device='cuda')
                del test_tensor
            except RuntimeError as e:
                print(f"CUDA初始化失败: {e}")
                print("切换到CPU")
                self.device = 'cpu'
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"使用设备: {self.device}")
        print(f"模型配置: d_model={self.d_model}, heads={self.num_heads}, layers={self.num_layers}")
