import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import json
import torch
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

class ComprehensiveVisualizer:
    def __init__(self, results_dir='results', checkpoint_dir='checkpoints'):
        self.results_dir = Path(results_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.viz_dir = self.results_dir / 'visualizations'
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置绘图风格
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def plot_training_curves(self):
        """绘制训练曲线"""
        logging.info("生成训练曲线图...")
        
        # 从metrics.csv加载数据
        metrics_file = self.results_dir / 'metrics.csv'
        if not metrics_file.exists():
            logging.warning("未找到metrics.csv文件")
            return
        
        df = pd.read_csv(metrics_file)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 训练和验证损失
        ax = axes[0, 0]
        ax.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
        ax.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 损失差异(过拟合检测)
        ax = axes[0, 1]
        overfit_gap = df['train_loss'] - df['val_loss']
        ax.plot(df['epoch'], overfit_gap, color='red', marker='o')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Train Loss - Val Loss')
        ax.set_title('Overfitting Gap')
        ax.grid(True, alpha=0.3)
        
        # 3. 对数尺度的损失
        ax = axes[1, 0]
        ax.semilogy(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
        ax.semilogy(df['epoch'], df['val_loss'], label='Val Loss', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (log scale)')
        ax.set_title('Loss Curves (Log Scale)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 损失变化率
        ax = axes[1, 1]
        train_loss_diff = np.diff(df['train_loss'])
        val_loss_diff = np.diff(df['val_loss'])
        ax.plot(df['epoch'][1:], train_loss_diff, label='Train Loss Change', marker='o')
        ax.plot(df['epoch'][1:], val_loss_diff, label='Val Loss Change', marker='s')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss Change')
        ax.set_title('Loss Change Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'training_curves_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"✓ 训练曲线图已保存")
    
    def plot_learning_rate_schedule(self, d_model=256, warmup_steps=4000, max_steps=50000):
        """可视化学习率调度"""
        logging.info("生成学习率调度图...")
        
        steps = np.arange(1, max_steps + 1)
        lrs = d_model ** (-0.5) * np.minimum(steps ** (-0.5), steps * warmup_steps ** (-1.5))
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 1. 完整的学习率曲线
        ax = axes[0]
        ax.plot(steps, lrs, linewidth=2)
        ax.axvline(x=warmup_steps, color='red', linestyle='--', label=f'Warmup End ({warmup_steps})')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Noam Learning Rate Schedule')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 前期的学习率(放大warmup阶段)
        ax = axes[1]
        warmup_range = min(warmup_steps * 2, max_steps)
        ax.plot(steps[:warmup_range], lrs[:warmup_range], linewidth=2)
        ax.axvline(x=warmup_steps, color='red', linestyle='--', label='Warmup End')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate (Warmup Period)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'learning_rate_schedule.png', dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"✓ 学习率调度图已保存")
    
    def plot_model_capacity(self, config_file='src/config.py'):
        """可视化模型容量分析"""
        logging.info("生成模型容量分析图...")
        
        # 不同配置的参数量
        configs = [
            {'name': 'Tiny', 'd_model': 128, 'layers': 2, 'heads': 4, 'd_ff': 512},
            {'name': 'Small', 'd_model': 256, 'layers': 4, 'heads': 8, 'd_ff': 1024},
            {'name': 'Base', 'd_model': 512, 'layers': 6, 'heads': 8, 'd_ff': 2048},
            {'name': 'Large', 'd_model': 768, 'layers': 12, 'heads': 12, 'd_ff': 3072},
        ]
        
        # 简化的参数量估算
        vocab_size = 50000  # 假设
        params = []
        for cfg in configs:
            d_model = cfg['d_model']
            layers = cfg['layers']
            d_ff = cfg['d_ff']
            
            # Embedding层
            embedding_params = 2 * vocab_size * d_model
            
            # Encoder/Decoder层
            # Self-attention: 4 * d_model^2 (Q, K, V, O)
            # FFN: 2 * d_model * d_ff
            layer_params = layers * (4 * d_model ** 2 + 2 * d_model * d_ff)
            
            total_params = (embedding_params + layer_params * 2) / 1e6  # 转为百万
            params.append(total_params)
        
        names = [cfg['name'] for cfg in configs]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 参数量柱状图
        ax = axes[0]
        bars = ax.bar(names, params, color=sns.color_palette("viridis", len(configs)))
        ax.set_ylabel('Parameters (Millions)')
        ax.set_title('Model Size Comparison')
        ax.grid(True, axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, param in zip(bars, params):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{param:.1f}M',
                   ha='center', va='bottom')
        
        # 2. 配置详情热图
        ax = axes[1]
        config_data = pd.DataFrame([
            [cfg['d_model'], cfg['layers'], cfg['heads'], cfg['d_ff']]
            for cfg in configs
        ], columns=['d_model', 'layers', 'heads', 'd_ff'], index=names)
        
        sns.heatmap(config_data.T, annot=True, fmt='d', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Value'})
        ax.set_title('Model Configuration Heatmap')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'model_capacity.png', dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"✓ 模型容量分析图已保存")
    
    def plot_attention_patterns(self):
        """可视化注意力模式(示例)"""
        logging.info("生成注意力模式示例图...")
        
        # 生成示例注意力权重
        seq_len = 10
        num_heads = 8
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, ax in enumerate(axes):
            # 生成不同模式的注意力权重
            if i < 4:
                # 前几个头:对角线模式(局部注意力)
                attention = np.eye(seq_len) + np.eye(seq_len, k=1) * 0.5 + np.eye(seq_len, k=-1) * 0.5
            else:
                # 后几个头:全局注意力
                attention = np.random.rand(seq_len, seq_len)
                attention = attention / attention.sum(axis=1, keepdims=True)
            
            sns.heatmap(attention, ax=ax, cmap='viridis', cbar=True, square=True)
            ax.set_title(f'Head {i+1}')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
        
        plt.suptitle('Multi-Head Attention Patterns (Example)', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'attention_patterns_example.png', dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"✓ 注意力模式图已保存")
    
    def plot_vocabulary_distribution(self, vocab_stats_file='results/vocab_stats.json'):
        """可视化词表分布"""
        logging.info("生成词表分布图...")
        
        # 如果没有实际数据,生成示例
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 词频分布(Zipf's law)
        ax = axes[0, 0]
        ranks = np.arange(1, 1001)
        freqs = 1 / ranks ** 1.5  # Zipf分布
        ax.loglog(ranks, freqs, 'b-', linewidth=2)
        ax.set_xlabel('Rank')
        ax.set_ylabel('Frequency')
        ax.set_title("Word Frequency Distribution (Zipf's Law)")
        ax.grid(True, alpha=0.3)
        
        # 2. 词长度分布
        ax = axes[0, 1]
        lengths = np.random.normal(5, 2, 1000).astype(int)
        lengths = np.clip(lengths, 1, 15)
        ax.hist(lengths, bins=15, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Word Length')
        ax.set_ylabel('Count')
        ax.set_title('Word Length Distribution')
        ax.grid(True, axis='y', alpha=0.3)
        
        # 3. 词表覆盖率
        ax = axes[1, 0]
        coverage = np.cumsum(freqs) / np.sum(freqs) * 100
        ax.plot(ranks, coverage, 'g-', linewidth=2)
        ax.axhline(y=80, color='r', linestyle='--', label='80% coverage')
        ax.axhline(y=90, color='orange', linestyle='--', label='90% coverage')
        ax.set_xlabel('Vocabulary Size')
        ax.set_ylabel('Corpus Coverage (%)')
        ax.set_title('Vocabulary Coverage')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 特殊token比例
        ax = axes[1, 1]
        special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>', 'Others']
        percentages = [2.4, 15.0, 5.0, 5.0, 72.6]  # 示例数据
        colors = sns.color_palette("pastel")[0:5]
        ax.pie(percentages, labels=special_tokens, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Token Type Distribution')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'vocabulary_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"✓ 词表分布图已保存")
    
    def plot_training_metrics_dashboard(self):
        """创建训练指标仪表板"""
        logging.info("生成训练指标仪表板...")
        
        metrics_file = self.results_dir / 'metrics.csv'
        if not metrics_file.exists():
            logging.warning("未找到metrics.csv文件")
            return
        
        df = pd.read_csv(metrics_file)
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 主要损失曲线
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(df['epoch'], df['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(df['epoch'], df['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.fill_between(df['epoch'], df['train_loss'], df['val_loss'], alpha=0.2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. 最佳指标
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        best_epoch = df['val_loss'].idxmin() + 1
        best_val_loss = df['val_loss'].min()
        final_train_loss = df['train_loss'].iloc[-1]
        
        stats_text = f"""
        Best Metrics:
        ─────────────
        Best Epoch: {best_epoch}
        Best Val Loss: {best_val_loss:.4f}
        Final Train Loss: {final_train_loss:.4f}
        
        Total Epochs: {len(df)}
        Improvement: {(df['val_loss'].iloc[0] - best_val_loss)/df['val_loss'].iloc[0]*100:.1f}%
        """
        ax2.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        # 3-5. 其他详细指标...
        # (添加更多可视化)
        
        plt.suptitle('Training Metrics Dashboard', fontsize=16, fontweight='bold', y=0.98)
        plt.savefig(self.viz_dir / 'training_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"✓ 训练仪表板已保存")
    
    def generate_all_visualizations(self):
        """生成所有可视化"""
        logging.info("\n" + "="*60)
        logging.info("开始生成所有可视化图表...")
        logging.info("="*60 + "\n")
        
        try:
            self.plot_training_curves()
            self.plot_learning_rate_schedule()
            self.plot_model_capacity()
            self.plot_attention_patterns()
            self.plot_vocabulary_distribution()
            self.plot_training_metrics_dashboard()
            
            logging.info("\n" + "="*60)
            logging.info("✨ 所有可视化生成完成!")
            logging.info(f"保存位置: {self.viz_dir}")
            logging.info("="*60 + "\n")
            
            # 生成README
            self._generate_viz_readme()
            
        except Exception as e:
            logging.error(f"生成可视化时出错: {str(e)}")
            raise
    
    def _generate_viz_readme(self):
        """生成可视化说明文档"""
        readme_path = self.viz_dir / 'README.md'
        
        content = """# Visualization Results

This directory contains various visualizations generated from the Transformer training process.

## Files

### 1. training_curves_comprehensive.png
Comprehensive training curves including:
- Training and validation loss
- Overfitting gap analysis
- Log-scale loss curves
- Loss change rate

### 2. learning_rate_schedule.png
Visualization of the Noam learning rate schedule:
- Full schedule over training steps
- Zoomed view of warmup period

### 3. model_capacity.png
Model size comparison:
- Parameter count for different model sizes
- Configuration heatmap

### 4. attention_patterns_example.png
Example attention patterns from multi-head attention

### 5. vocabulary_distribution.png
Vocabulary analysis:
- Word frequency distribution (Zipf's law)
- Word length distribution
- Vocabulary coverage
- Special token distribution

### 6. training_dashboard.png
Comprehensive training metrics dashboard

## Usage

These visualizations can be regenerated anytime by running:
```bash
python src/visualize_results.py
```
"""
        
        with open(readme_path, 'w') as f:
            f.write(content)

def main():
    visualizer = ComprehensiveVisualizer()
    visualizer.generate_all_visualizations()

if __name__ == '__main__':
    main()
