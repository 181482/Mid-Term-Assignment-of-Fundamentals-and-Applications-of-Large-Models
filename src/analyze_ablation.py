import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def fetch_wandb_results():
    """从wandb获取所有消融实验的结果"""
    api = wandb.Api()
    runs = api.runs("322-beijing-jiaotong-university/transformer-ablation")
    
    results = []
    for run in runs:
        if run.name != 'summary':  # 排除汇总运行
            try:
                history = run.history()
                if not history.empty:
                    val_losses = history['val_loss'].dropna()
                    train_losses = history['train_loss'].dropna()
                    
                    if not val_losses.empty and not train_losses.empty:
                        results.append({
                            'experiment': run.name,
                            'val_loss': val_losses.min(),
                            'final_train_loss': train_losses.iloc[-1],
                            'epochs': len(train_losses),
                            'config': run.config
                        })
                    
            except Exception as e:
                print(f"警告: 处理运行 {run.name} 时出错: {str(e)}")
                continue
    
    if not results:
        raise ValueError("没有找到有效的实验结果")
        
    return pd.DataFrame(results)

def plot_ablation_results(df, save_dir='results/ablation'):
    """绘制消融实验结果的对比图"""
    os.makedirs(save_dir, exist_ok=True)
    
    if df.empty:
        print("警告: 没有数据可以绘图")
        return
        
    # 设置风格
    plt.style.use('seaborn')
    
    # 1. 验证损失对比图
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='experiment', y='val_loss')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Experiment Configuration')
    plt.ylabel('Validation Loss')
    plt.ylim(0.145, 0.155)  # 设置y轴范围
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'val_loss_comparison.png'))
    plt.close()
    
    # 2. 训练损失vs验证损失
    plt.figure(figsize=(10, 6))
    plt.scatter(df['final_train_loss'], df['val_loss'])
    for i, txt in enumerate(df['experiment']):
        plt.annotate(txt, (df['final_train_loss'].iloc[i], df['val_loss'].iloc[i]))
    plt.xlabel('Final Training Loss')
    plt.ylabel('Best Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'train_vs_val_loss.png'))
    plt.close()
    
    # 3. 实验结果汇总表
    summary = df.sort_values('val_loss')
    summary.to_csv(os.path.join(save_dir, 'ablation_summary.csv'), index=False)
    
    # 4. 创建详细的实验报告
    with open(os.path.join(save_dir, 'ablation_report.md'), 'w') as f:
        f.write('# Transformer Ablation Study Report\n\n')
        f.write('## Results Summary\n\n')
        f.write(summary.to_markdown())
        f.write('\n\n## Key Findings\n\n')
        
        best_config = summary.iloc[0]
        f.write(f"- Best Configuration: {best_config['experiment']}\n")
        f.write(f"- Best Validation Loss: {best_config['val_loss']:.4f}\n")
        
        heads_results = df[df['experiment'].str.contains('heads_')]
        f.write('\n### Impact of Attention Heads\n')
        f.write(heads_results.to_markdown())
        
        layers_results = df[df['experiment'].str.contains('layers_')]
        f.write('\n### Impact of Model Depth\n')
        f.write(layers_results.to_markdown())

def main():
    try:
        print("正在从wandb获取实验结果...")
        results = fetch_wandb_results()
        print(f"成功获取到 {len(results)} 个实验结果")
        
        print("正在生成分析图表...")
        plot_ablation_results(results)
        
        print("分析完成！结果保存在 results/ablation 目录下")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        print("如果问题持续存在，请检查wandb项目配置和实验运行状态")

if __name__ == '__main__':
    main()
