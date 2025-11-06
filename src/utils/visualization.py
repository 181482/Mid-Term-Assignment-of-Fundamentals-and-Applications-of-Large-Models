import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

class Visualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_training_curves(self, train_losses, val_losses, epochs):
        if len(train_losses) == 0 or len(val_losses) == 0:
            print("Warning: No loss data to plot")
            return
            
        if len(train_losses) != len(epochs) or len(val_losses) != len(epochs):
            print(f"Warning: Length mismatch - epochs: {len(epochs)}, train_losses: {len(train_losses)}, val_losses: {len(val_losses)}")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label='Training Loss')
        plt.plot(epochs, val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'))
        plt.close()
        
    def plot_attention_weights(self, attention_weights, src_tokens, tgt_tokens):
        plt.figure(figsize=(10, 10))
        sns.heatmap(
            attention_weights,
            xticklabels=src_tokens,
            yticklabels=tgt_tokens,
            cmap='viridis'
        )
        plt.title('Attention Weights Visualization')
        plt.savefig(os.path.join(self.save_dir, 'attention_weights.png'))
        plt.close()
        
    def save_metrics_to_csv(self, metrics_dict):
        df = pd.DataFrame(metrics_dict)
        df.to_csv(os.path.join(self.save_dir, 'metrics.csv'), index=False)
