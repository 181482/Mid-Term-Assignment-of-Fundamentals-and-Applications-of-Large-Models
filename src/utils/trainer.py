import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import wandb
import os
from torch.cuda.amp import autocast, GradScaler
from sacrebleu.metrics import BLEU
import math
import logging
import sys

class NoamLR:
    """Noam学习率调度器"""
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def get_lr(self):
        step = self.current_step
        return self.d_model ** (-0.5) * min(step ** (-0.5), 
                                          step * self.warmup_steps ** (-1.5))
    
    def state_dict(self):
        """返回调度器状态"""
        return {
            'current_step': self.current_step,
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps
        }
    
    def load_state_dict(self, state_dict):
        """加载调度器状态"""
        self.current_step = state_dict['current_step']
        self.d_model = state_dict['d_model']
        self.warmup_steps = state_dict['warmup_steps']

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config
    ):
        self.device = config.device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.train_losses = []  # 初始化损失记录列表
        self.val_losses = []
        
        self.criterion = nn.CrossEntropyLoss()
        
        # 初始化评估指标
        self.bleu = BLEU()
        self.steps = 0  # 用于追踪总训练步数
        self.best_metrics = {'val_loss': float('inf'), 'bleu': 0, 'ppl': float('inf')}
        
        # 设置优化器
        if config.optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        
        # 使用Noam学习率调度器
        self.scheduler = NoamLR(
            self.optimizer, 
            config.d_model, 
            config.warmup_steps
        )
        
        # 只在CUDA设备上使用混合精度训练
        self.use_amp = self.device == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        self.accumulation_steps = config.accumulation_steps  # 梯度累积步数

        # 设置torch的内存分配器
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            # 限制缓存分配器
            torch.cuda.set_per_process_memory_fraction(0.8)
        
        # 保存数据加载器的词表引用
        self.src_vocab = train_loader.dataset.src_vocab
        self.tgt_vocab = train_loader.dataset.tgt_vocab

        # 添加早停计数器
        self.patience_counter = 0
        self.best_val_loss = float('inf')

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        # 计算总batch数和10%的间隔
        total_batches = len(self.train_loader)
        log_interval = max(1, total_batches // 10)  # 每10%显示一次
        
        for i, batch in enumerate(self.train_loader):
            # 定期清理缓存
            if i % 100 == 0 and self.device == 'cuda':
                torch.cuda.empty_cache()
            
            src, tgt = batch
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            # 在累积步骤开始时清零梯度
            if i % self.accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            # CPU或CUDA的不同训练策略
            if self.use_amp:
                with autocast():
                    output = self.model(src, tgt[:, :-1])
                    loss = self.criterion(
                        output.reshape(-1, output.size(-1)),
                        tgt[:, 1:].reshape(-1)
                    )
                    loss = loss / self.accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            else:
                output = self.model(src, tgt[:, :-1])
                loss = self.criterion(
                    output.reshape(-1, output.size(-1)),
                    tgt[:, 1:].reshape(-1)
                )
                loss = loss / self.accumulation_steps
                
                loss.backward()
                
                if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            
            total_loss += loss.item() * self.accumulation_steps
            self.steps += 1
            
            # 每10%显示一次进度
            if (i + 1) % log_interval == 0 or (i + 1) == total_batches:
                progress = (i + 1) / total_batches * 100
                current_loss = loss.item() * self.accumulation_steps
                lr = self.optimizer.param_groups[0]['lr']
                print(f'\rTraining: {progress:.0f}% [{i+1}/{total_batches}] | Loss: {current_loss:.4f} | LR: {lr:.2e}', 
                      end='', flush=True)
            
            # 只记录到wandb
            if self.steps % 100 == 0 and wandb.run is not None:
                wandb.log({
                    'train_loss': loss.item() * self.accumulation_steps,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'step': self.steps
                })
        
        print()  # 换行
        return total_loss / len(self.train_loader)
    
    def decode_tokens(self, tokens):
        """将token id序列转换为文本
        
        Args:
            tokens: tensor或numpy array,包含token ids
            
        Returns:
            str: 解码后的文本字符串
        """
        words = []
        for token in tokens:
            idx = token.item() if torch.is_tensor(token) else int(token)
            
            # 遇到pad就停止
            if idx == self.tgt_vocab['<pad>']:
                break
            
            # 跳过特殊标记
            if idx in [self.tgt_vocab['<bos>'], self.tgt_vocab['<eos>']]:
                continue
            
            try:
                word = self.tgt_vocab.get_itos()[idx]
                words.append(word)
            except IndexError:
                logging.warning(f"Token ID {idx} 超出词表范围")
                continue
                
        return ''.join(words)  # 中文不需要空格
    
    def calculate_bleu(self, num_samples=100):
        """计算BLEU分数
        
        BLEU (BiLingual Evaluation Understudy):
        - 通过n-gram匹配度评估翻译质量
        - 分数范围: 0-100 (sacrebleu) 或 0-1
        - 越高表示与参考翻译越相似
        
        关键点:
        1. 使用自回归生成获取模型翻译(而非teacher forcing)
        2. 中文BLEU: 字符级别或词级别
        3. sacrebleu格式: hypotheses为字符串列表, references为列表的列表
        
        Args:
            num_samples: 用于计算BLEU的样本数量
            
        Returns:
            float: BLEU分数 (0-100)
        """
        self.model.eval()
        hypotheses = []  # 模型翻译列表
        references = []   # 参考翻译列表 (每个元素是一个列表,支持多参考)
        
        logging.info(f"开始计算BLEU分数 (样本数: {num_samples})...")
        
        with torch.no_grad():
            sample_count = 0
            
            for batch in self.val_loader:
                if sample_count >= num_samples:
                    break
                    
                src, tgt = batch
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                
                batch_size = src.size(0)
                
                for i in range(batch_size):
                    if sample_count >= num_samples:
                        break
                    
                    # 单个样本
                    src_i = src[i:i+1]
                    tgt_i = tgt[i:i+1]
                    
                    # 1. 获取参考翻译(ground truth)
                    ref_tokens = []
                    for token_id in tgt_i[0].cpu().numpy():
                        if token_id == self.tgt_vocab['<pad>']:
                            break
                        if token_id in [self.tgt_vocab['<bos>'], self.tgt_vocab['<eos>']]:
                            continue
                        try:
                            word = self.tgt_vocab.get_itos()[token_id]
                            ref_tokens.append(word)
                        except:
                            continue
                    
                    ref_text = ''.join(ref_tokens)
                    
                    # 2. 自回归生成模型翻译
                    generated_ids = [self.tgt_vocab['<bos>']]
                    
                    for _ in range(self.config.max_seq_length - 1):
                        tgt_input = torch.tensor([generated_ids], dtype=torch.long).to(self.device)
                        
                        try:
                            output = self.model(src_i, tgt_input)
                            next_token = output[0, -1, :].argmax().item()
                            generated_ids.append(next_token)
                            
                            if next_token == self.tgt_vocab['<eos>']:
                                break
                        except:
                            break
                    
                    # 解码生成的翻译
                    pred_tokens = []
                    for token_id in generated_ids[1:]:  # 跳过<bos>
                        if token_id == self.tgt_vocab['<eos>']:
                            break
                        try:
                            word = self.tgt_vocab.get_itos()[token_id]
                            pred_tokens.append(word)
                        except:
                            continue
                    
                    pred_text = ''.join(pred_tokens)
                    
                    # 只添加非空翻译
                    if pred_text.strip() and ref_text.strip():
                        # 中文BLEU: 字符级别评估
                        # 在每个字符间添加空格,这样sacrebleu可以计算字符级n-gram
                        pred_chars = ' '.join(list(pred_text))
                        ref_chars = ' '.join(list(ref_text))
                        
                        hypotheses.append(pred_chars)
                        references.append([ref_chars])  # sacrebleu需要列表的列表
                        sample_count += 1
        
        if len(hypotheses) == 0:
            logging.warning("没有有效的翻译样本用于BLEU计算")
            return 0.0
        
        try:
            # 使用sacrebleu计算BLEU
            # sacrebleu.corpus_score 返回 0-100 的分数
            from sacrebleu.metrics import BLEU
            bleu_metric = BLEU()
            result = bleu_metric.corpus_score(hypotheses, references)
            
            logging.info(f"BLEU计算完成: {result.score:.2f} (样本数: {len(hypotheses)})")
            
            # 打印样例用于验证
            if len(hypotheses) > 0 and logging.getLogger().level == logging.DEBUG:
                logging.debug(f"样例模型翻译: {hypotheses[0][:100]}")
                logging.debug(f"样例参考翻译: {references[0][0][:100]}")
            
            # 返回0-100范围的BLEU分数
            return result.score
            
        except Exception as e:
            logging.error(f"计算BLEU时出错: {str(e)}")
            logging.error(f"hypotheses示例: {hypotheses[0] if hypotheses else 'N/A'}")
            logging.error(f"references示例: {references[0] if references else 'N/A'}")
            return 0.0
    
    def calculate_perplexity(self, loss):
        """计算困惑度
        
        Args:
            loss: 交叉熵损失
            
        Returns:
            float: 困惑度值
        """
        try:
            # 限制最大值避免溢出
            ppl = math.exp(min(loss, 100))
            return ppl
        except:
            return float('inf')
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        total_val_batches = len(self.val_loader)
        log_interval = max(1, total_val_batches // 10)
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                src, tgt = batch
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                
                # 使用teacher forcing计算损失
                output = self.model(src, tgt[:, :-1])
                loss = self.criterion(
                    output.reshape(-1, output.size(-1)),
                    tgt[:, 1:].reshape(-1)
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                # 每10%显示一次进度
                if (i + 1) % log_interval == 0 or (i + 1) == total_val_batches:
                    progress = (i + 1) / total_val_batches * 100
                    print(f'\rValidating: {progress:.0f}% [{i+1}/{total_val_batches}]', 
                          end='', flush=True)
        
        print()  # 换行
        
        avg_loss = total_loss / num_batches
        
        # 计算困惑度
        avg_ppl = self.calculate_perplexity(avg_loss)
        
        # 计算BLEU (较慢,使用子集)
        bleu_score = self.calculate_bleu(num_samples=50)
        
        return {
            'val_loss': avg_loss,
            'ppl': avg_ppl,
            'bleu': bleu_score
        }
    
    def train(self):
        for epoch in range(1, self.config.epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{self.config.epochs}")
            print(f"{'='*50}")
            
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            
            # 记录损失值
            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['val_loss'])
            
            # 记录到wandb
            if wandb.run is not None:
                wandb.log({
                    'epoch': epoch,
                    'epoch_train_loss': train_loss,
                    'epoch_val_loss': val_metrics['val_loss'],
                    'epoch_val_ppl': val_metrics['ppl'],
                    'epoch_val_bleu': val_metrics['bleu']
                })
            
            # 每个epoch保存一次模型
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(f'best_model_epoch_{epoch}.pth')
                self.patience_counter = 0
                print(f"✓ 保存最佳模型 (epoch {epoch})")
            else:
                self.patience_counter += 1
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val PPL: {val_metrics['ppl']:.2f}")
            print(f"  Val BLEU: {val_metrics['bleu']:.2f}")
            
            # 早停检查
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\n早停触发！连续{self.config.early_stopping_patience}个epoch未改善")
                break
        
        print(f"\n训练完成！最佳验证损失: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, filename):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': len(self.train_losses),
            'best_val_loss': self.best_val_loss,
            'steps': self.steps
        }
        path = os.path.join(self.config.checkpoint_dir, filename)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, filename):
        """加载checkpoint"""
        path = os.path.join(self.config.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.steps = checkpoint.get('steps', 0)
        
        return checkpoint.get('epoch', 0)

