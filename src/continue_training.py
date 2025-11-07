import torch
import random
import numpy as np
import argparse
import logging
import os
from models.transformer import Transformer
from utils.data_loader import DataModule
from utils.trainer import Trainer
from utils.visualization import Visualizer
from config import TransformerConfig
import wandb

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

def find_latest_checkpoint(checkpoint_dir):
    """查找最新的checkpoint文件"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                       if f.startswith('best_model') and f.endswith('.pth')]
    
    if not checkpoint_files:
        return None
    
    # 按修改时间排序,选择最新的
    checkpoint_files.sort(
        key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)),
        reverse=True
    )
    
    return os.path.join(checkpoint_dir, checkpoint_files[0])

def main():
    parser = argparse.ArgumentParser(description='继续训练Transformer模型')
    parser.add_argument('--checkpoint', type=str, help='checkpoint文件路径,如果不指定则自动找最新的')
    parser.add_argument('--epochs', type=int, default=20, help='总共要训练的epoch数')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='强制使用指定设备')
    args = parser.parse_args()
    
    # 加载配置
    config = TransformerConfig()
    if args.device:
        config.device = args.device
    
    # 禁用早停 - 设置一个很大的patience值
    config.early_stopping_patience = 999999
    config.epochs = args.epochs
    
    config.__post_init__()
    set_seed(config.seed)
    
    logging.info("="*60)
    logging.info("继续训练模式")
    logging.info(f"目标总epoch数: {config.epochs}")
    logging.info(f"早停已禁用")
    logging.info(f"BLEU计算已禁用(加快训练)")
    logging.info("="*60)
    
    try:
        # 查找checkpoint
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = find_latest_checkpoint(config.checkpoint_dir)
        
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            logging.error("未找到checkpoint文件!")
            logging.info("请先运行初始训练或指定checkpoint路径:")
            logging.info("  python src/continue_training.py --checkpoint checkpoints/best_model_epoch_3.pth")
            return
        
        logging.info(f"找到checkpoint: {checkpoint_path}")
        
        # 加载checkpoint获取模型配置
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        
        # 从checkpoint推断模型配置
        d_model = checkpoint['model_state_dict']['encoder_embedding.weight'].shape[1]
        max_seq_length = checkpoint['model_state_dict']['positional_encoding.pe'].shape[1]
        num_layers = sum(1 for k in checkpoint['model_state_dict'].keys() 
                        if k.startswith('encoder.layers.') and k.endswith('.norm1.weight'))
        
        if d_model == 128:
            num_heads = 4
        elif d_model == 256:
            num_heads = 8
        else:
            num_heads = 8
        
        # 获取当前已训练的epoch数
        current_epoch = checkpoint.get('epoch', 0)
        
        logging.info(f"检测到的模型配置:")
        logging.info(f"  d_model: {d_model}")
        logging.info(f"  num_heads: {num_heads}")
        logging.info(f"  num_layers: {num_layers}")
        logging.info(f"  max_seq_length: {max_seq_length}")
        logging.info(f"  已完成epoch: {current_epoch}")
        logging.info(f"  将继续训练到epoch: {config.epochs}")
        
        if current_epoch >= config.epochs:
            logging.warning(f"模型已训练{current_epoch}个epoch,达到或超过目标{config.epochs}个epoch")
            response = input("是否继续训练更多epoch? (y/n): ")
            if response.lower() != 'y':
                return
        
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
        
        train_loader, val_loader, test_loader = data.get_dataloaders()
        
        logging.info("="*60)
        logging.info("初始化模型...")
        logging.info("="*60)
        
        # 创建模型
        model = Transformer(
            src_vocab_size=len(data.src_vocab),
            tgt_vocab_size=len(data.tgt_vocab),
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            max_seq_length=max_seq_length
        ).to(config.device)
        
        # 初始化训练器
        trainer = Trainer(model, train_loader, val_loader, config)
        
        # 临时修改trainer的validate方法,跳过BLEU计算
        original_validate = trainer.validate
        
        def fast_validate():
            """快速验证,不计算BLEU"""
            trainer.model.eval()
            total_loss = 0.0
            num_batches = 0
            
            total_val_batches = len(trainer.val_loader)
            log_interval = max(1, total_val_batches // 10)
            
            with torch.no_grad():
                for i, batch in enumerate(trainer.val_loader):
                    src, tgt = batch
                    src = src.to(trainer.device)
                    tgt = tgt.to(trainer.device)
                    
                    output = trainer.model(src, tgt[:, :-1])
                    loss = trainer.criterion(
                        output.reshape(-1, output.size(-1)),
                        tgt[:, 1:].reshape(-1)
                    )
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    if (i + 1) % log_interval == 0 or (i + 1) == total_val_batches:
                        progress = (i + 1) / total_val_batches * 100
                        print(f'\rValidating: {progress:.0f}% [{i+1}/{total_val_batches}]', 
                              end='', flush=True)
            
            print()
            
            avg_loss = total_loss / num_batches
            avg_ppl = trainer.calculate_perplexity(avg_loss)
            
            return {
                'val_loss': avg_loss,
                'ppl': avg_ppl,
                'bleu': 0.0  # 不计算BLEU
            }
        
        # 替换validate方法
        trainer.validate = fast_validate
        
        logging.info("加载checkpoint...")
        loaded_epoch = trainer.load_checkpoint(os.path.basename(checkpoint_path))
        
        logging.info(f"✓ 成功加载checkpoint")
        logging.info(f"  当前epoch: {loaded_epoch}")
        logging.info(f"  当前最佳验证损失: {trainer.best_val_loss:.4f}")
        logging.info(f"  训练步数: {trainer.steps}")
        
        # 初始化wandb
        wandb.init(
            project='transformer-translation',
            name=f'continue-from-epoch-{loaded_epoch}',
            config=vars(config),
            resume='allow'
        )
        
        logging.info("="*60)
        logging.info(f"继续训练 (从epoch {loaded_epoch + 1} 到 {config.epochs})...")
        logging.info("="*60)
        
        # 修改trainer的epoch范围
        remaining_epochs = config.epochs - loaded_epoch
        logging.info(f"剩余需要训练的epoch数: {remaining_epochs}")
        
        if remaining_epochs <= 0:
            logging.info("已达到目标epoch数,不需要继续训练")
            wandb.finish()
            return
        
        # 手动训练剩余的epochs
        visualizer = Visualizer(config.results_dir)
        
        for epoch in range(loaded_epoch + 1, config.epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{config.epochs}")
            print(f"{'='*50}")
            
            train_loss = trainer.train_epoch()
            val_metrics = trainer.validate()
            
            # 记录损失值
            trainer.train_losses.append(train_loss)
            trainer.val_losses.append(val_metrics['val_loss'])
            
            # 记录到wandb
            wandb.log({
                'epoch': epoch,
                'epoch_train_loss': train_loss,
                'epoch_val_loss': val_metrics['val_loss'],
                'epoch_val_ppl': val_metrics['ppl'],
                'epoch_val_bleu': val_metrics['bleu']
            })
            
            # 保存最佳模型
            if val_metrics['val_loss'] < trainer.best_val_loss:
                trainer.best_val_loss = val_metrics['val_loss']
                trainer.save_checkpoint(f'best_model_epoch_{epoch}.pth')
                print(f"✓ 保存最佳模型 (epoch {epoch})")
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val PPL: {val_metrics['ppl']:.2f}")
            # print(f"  Val BLEU: {val_metrics['bleu']:.2f}")  # 注释掉BLEU显示
        
        logging.info("="*60)
        logging.info("保存最终结果...")
        logging.info("="*60)
        
        # 保存训练曲线
        if len(trainer.train_losses) > 0:
            epochs = list(range(loaded_epoch + 1, loaded_epoch + 1 + len(trainer.train_losses)))
            visualizer.plot_training_curves(
                trainer.train_losses,
                trainer.val_losses,
                epochs
            )
            
            # 保存指标(追加模式)
            metrics = {
                'epoch': epochs,
                'train_loss': trainer.train_losses,
                'val_loss': trainer.val_losses
            }
            
            # 如果已有metrics文件,则追加
            import pandas as pd
            metrics_file = os.path.join(config.results_dir, 'metrics.csv')
            if os.path.exists(metrics_file):
                old_df = pd.read_csv(metrics_file)
                new_df = pd.DataFrame(metrics)
                combined_df = pd.concat([old_df, new_df], ignore_index=True)
                combined_df.to_csv(metrics_file, index=False)
            else:
                visualizer.save_metrics_to_csv(metrics)
        
        wandb.finish()
        
        print(f"\n{'='*60}")
        print(f"✨ 训练完成!")
        print(f"  总训练epoch: {config.epochs}")
        print(f"  最佳验证损失: {trainer.best_val_loss:.4f}")
        print(f"  模型保存在: {config.checkpoint_dir}")
        print(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        logging.info("\n训练被中断")
        wandb.finish()
    except Exception as e:
        logging.error(f"训练过程中出错: {str(e)}", exc_info=True)
        wandb.finish()
        raise

if __name__ == '__main__':
    main()
