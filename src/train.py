import torch
import random
import numpy as np
import argparse
import logging
from models.transformer import Transformer
from utils.data_loader import DataModule
from utils.trainer import Trainer
from utils.visualization import Visualizer
from config import TransformerConfig
import wandb

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='强制使用指定设备')
    parser.add_argument('--resume', type=str, help='从checkpoint恢复训练')
    args = parser.parse_args()
    
    # 加载配置
    config = TransformerConfig()
    if args.device:
        config.device = args.device
    config.__post_init__()
    set_seed(config.seed)
    
    try:
        logging.info("="*60)
        logging.info("初始化数据加载器...")
        logging.info("="*60)
        
        # 初始化数据加载器
        data = DataModule(
            batch_size=config.batch_size,
            max_length=config.max_seq_length,
            config=config
        )
        data.setup()
        
        # 数据集统计
        logging.info(f"源语言词表大小: {len(data.src_vocab)}")
        logging.info(f"目标语言词表大小: {len(data.tgt_vocab)}")
        
        # 获取数据加载器
        train_loader, val_loader, test_loader = data.get_dataloaders()
        
        # 检查数据
        src, tgt = next(iter(train_loader))
        logging.info(f"批次形状 - src: {src.shape}, tgt: {tgt.shape}")
        logging.info(f"最大token ID - src: {src.max()}, tgt: {tgt.max()}")
        
        # 验证token ID在词表范围内
        assert src.max() < len(data.src_vocab), f"源语言token ID超出范围: {src.max()} >= {len(data.src_vocab)}"
        assert tgt.max() < len(data.tgt_vocab), f"目标语言token ID超出范围: {tgt.max()} >= {len(data.tgt_vocab)}"
        
        logging.info("="*60)
        logging.info("初始化模型...")
        logging.info("="*60)
        
        # 初始化模型
        model = Transformer(
            src_vocab_size=len(data.src_vocab),
            tgt_vocab_size=len(data.tgt_vocab),
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            max_seq_length=config.max_seq_length
        ).to(config.device)
        
        # 打印模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"总参数量: {total_params:,}")
        logging.info(f"可训练参数: {trainable_params:,}")
        
        # 初始化训练器
        trainer = Trainer(model, train_loader, val_loader, config)
        visualizer = Visualizer(config.results_dir)
        
        # 初始化wandb
        wandb.init(
            project='transformer-translation',
            name=f'iwslt17-zh-en-{config.d_model}d-{config.num_layers}l',
            config=vars(config)
        )
        
        logging.info("="*60)
        logging.info("开始训练...")
        logging.info("="*60)
        
        # 开始训练
        trainer.train()
        
        logging.info("="*60)
        logging.info("保存结果...")
        logging.info("="*60)
        
        # 保存训练曲线
        if len(trainer.train_losses) > 0:
            epochs = list(range(1, len(trainer.train_losses) + 1))
            visualizer.plot_training_curves(
                trainer.train_losses,
                trainer.val_losses,
                epochs
            )
            
            # 保存指标
            metrics = {
                'epoch': epochs,
                'train_loss': trainer.train_losses,
                'val_loss': trainer.val_losses
            }
            visualizer.save_metrics_to_csv(metrics)
        
        wandb.finish()
        logging.info("训练完成！")
        
    except KeyboardInterrupt:
        logging.info("\n训练被中断")
        wandb.finish()
    except Exception as e:
        logging.error(f"训练过程中出错: {str(e)}", exc_info=True)
        wandb.finish()
        raise

if __name__ == '__main__':
    main()
