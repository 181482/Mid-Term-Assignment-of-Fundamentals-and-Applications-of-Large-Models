import torch
import pandas as pd
from models.transformer import Transformer
from utils.data_loader import DataModule
from utils.trainer import Trainer
from config import TransformerConfig
import wandb
import logging

logging.basicConfig(level=logging.INFO)

def run_experiment(config, experiment_name, **kwargs):
    """运行单个消融实验"""
    # 提取位置编码参数
    use_pe = kwargs.pop('use_positional_encoding', True)
    
    # 更新其他配置
    for key, value in kwargs.items():
        setattr(config, key, value)
    
    # 初始化wandb
    wandb.init(
        project='transformer-ablation',
        name=experiment_name,
        config=vars(config),
        reinit=True
    )
    
    try:
        # 初始化数据
        data = DataModule(
            batch_size=config.batch_size,
            max_length=config.max_seq_length,
            config=config
        )
        data.setup()
        
        # 获取数据加载器 - 现在返回3个值
        train_loader, val_loader, test_loader = data.get_dataloaders()
        
        # 初始化模型
        model = Transformer(
            src_vocab_size=len(data.src_vocab),
            tgt_vocab_size=len(data.tgt_vocab),
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            max_seq_length=config.max_seq_length,
            use_positional_encoding=use_pe  # 传递位置编码参数
        ).to(config.device)
        
        # 训练模型
        trainer = Trainer(model, train_loader, val_loader, config)
        trainer.train()
        
        best_loss = trainer.best_val_loss
        
    except Exception as e:
        logging.error(f"实验 {experiment_name} 失败: {str(e)}")
        best_loss = float('inf')
    finally:
        # 结束当前实验
        wandb.finish()
    
    return best_loss

def main():
    base_config = TransformerConfig()
    base_config.epochs = 5  # 消融实验使用较少的epoch
    results = []
    
    logging.info("开始消融实验...")
    
    # Baseline实验
    logging.info("\n" + "="*50)
    logging.info("运行 Baseline 实验")
    logging.info("="*50)
    results.append({
        'experiment': 'baseline',
        'config': f'd_model={base_config.d_model}, heads={base_config.num_heads}, layers={base_config.num_layers}',
        'val_loss': run_experiment(base_config.copy() if hasattr(base_config, 'copy') else TransformerConfig(), 'baseline')
    })


    # 位置编码消融实验
    logging.info("\n" + "="*50)
    logging.info("运行实验: 无位置编码")
    logging.info("="*50)
    config = TransformerConfig()
    config.epochs = 5
    results.append({
        'experiment': 'no_positional_encoding',
        'config': 'without positional encoding',
        'val_loss': run_experiment(config, 'no_positional_encoding', use_positional_encoding=False)
    })
    
    # 不同头数的实验
    for num_heads in [2, 4, 8]:
        logging.info("\n" + "="*50)
        logging.info(f"运行实验: num_heads={num_heads}")
        logging.info("="*50)
        config = TransformerConfig()
        config.epochs = 5
        results.append({
            'experiment': f'heads_{num_heads}',
            'config': f'heads={num_heads}',
            'val_loss': run_experiment(config, f'heads_{num_heads}', num_heads=num_heads)
        })
    
    # 不同层数的实验
    for num_layers in [2, 4, 6]:
        logging.info("\n" + "="*50)
        logging.info(f"运行实验: num_layers={num_layers}")
        logging.info("="*50)
        config = TransformerConfig()
        config.epochs = 5
        results.append({
            'experiment': f'layers_{num_layers}',
            'config': f'layers={num_layers}',
            'val_loss': run_experiment(config, f'layers_{num_layers}', num_layers=num_layers)
        })
    
   
    
    
    
    # 保存结果
    import os
    os.makedirs('results/ablation', exist_ok=True)
    
    df = pd.DataFrame(results)
    df = df.sort_values('val_loss')
    df.to_csv('results/ablation/ablation_results.csv', index=False)
    
    # 打印结果摘要
    logging.info("\n" + "="*50)
    logging.info("消融实验完成！结果摘要:")
    logging.info("="*50)
    print(df.to_string(index=False))
    
    # 创建最终的汇总运行
    wandb.init(
        project='transformer-ablation',
        name='summary',
        reinit=True
    )
    wandb.log({'ablation_results': wandb.Table(dataframe=df)})
    wandb.finish()
    
    logging.info(f"\n详细结果已保存到: results/ablation/ablation_results.csv")
    logging.info(f"最佳配置: {df.iloc[0]['experiment']} (val_loss={df.iloc[0]['val_loss']:.4f})")

if __name__ == '__main__':
    main()
