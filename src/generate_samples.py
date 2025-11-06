import torch
import argparse
from models.transformer import Transformer
from utils.data_loader import DataModule
from config import TransformerConfig
import os
import logging

logging.basicConfig(level=logging.INFO)

class TranslationGenerator:
    def __init__(self, model, src_vocab, tgt_vocab, device='cuda'):
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        self.max_length = 128
        
    def translate(self, src_text, src_tokenizer, temperature=1.0):
        """翻译文本,添加温度采样"""
        self.model.eval()
        
        try:
            with torch.no_grad():
                # 分词并编码
                src_tokens = src_tokenizer.tokenize(src_text)
                
                # 限制源文本长度
                if len(src_tokens) > self.max_length - 2:
                    src_tokens = src_tokens[:self.max_length - 2]
                
                src_ids = [self.src_vocab['<bos>']] + \
                         [self.src_vocab[t] for t in src_tokens] + \
                         [self.src_vocab['<eos>']]
                
                # 填充
                src_len = len(src_ids)
                src_ids += [self.src_vocab['<pad>']] * (self.max_length - len(src_ids))
                src_ids = src_ids[:self.max_length]
                
                src = torch.tensor([src_ids], dtype=torch.long).to(self.device)
                
                # 确保tensor是连续的
                src = src.contiguous()
                
                # 初始化目标序列
                tgt = torch.tensor([[self.tgt_vocab['<bos>']]], dtype=torch.long).to(self.device)
                
                # 逐token生成
                for step in range(self.max_length - 1):
                    # 确保tgt也是连续的
                    tgt = tgt.contiguous()
                    
                    # 前向传播
                    try:
                        output = self.model(src, tgt)
                    except RuntimeError as e:
                        logging.error(f"模型前向传播错误: {str(e)}")
                        logging.error(f"src shape: {src.shape}, tgt shape: {tgt.shape}")
                        return ""
                    
                    # 获取最后一个token的预测
                    next_token_logits = output[:, -1, :] / temperature
                    
                    # 使用top-k采样而不是贪婪搜索
                    top_k = 10
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    probs = torch.softmax(top_k_logits, dim=-1)
                    
                    # 采样下一个token
                    sampled_index = torch.multinomial(probs, 1)
                    next_token = top_k_indices.gather(1, sampled_index)
                    
                    # 记录生成的token(用于调试)
                    if step < 5:  # 只记录前5个token
                        token_id = next_token.item()
                        token_word = self.tgt_vocab.get_itos()[token_id] if token_id < len(self.tgt_vocab.get_itos()) else '<invalid>'
                        logging.debug(f"Step {step}: token_id={token_id}, word={token_word}")
                    
                    # 添加到目标序列
                    tgt = torch.cat([tgt, next_token], dim=1)
                    
                    # 如果生成了结束符或达到最大长度，停止
                    if next_token.item() == self.tgt_vocab['<eos>'] or tgt.size(1) >= self.max_length:
                        break
                
                # 解码
                tgt_ids = tgt[0].cpu().numpy()
                translated_words = []
                for idx in tgt_ids:
                    if idx == self.tgt_vocab['<pad>']:
                        break
                    if idx in [self.tgt_vocab['<bos>'], self.tgt_vocab['<eos>']]:
                        continue
                    try:
                        word = self.tgt_vocab.get_itos()[idx]
                        translated_words.append(word)
                    except Exception as e:
                        logging.warning(f"解码token {idx} 时出错: {str(e)}")
                        continue
                
                return ''.join(translated_words)  # 中文不需要空格
                
        except Exception as e:
            logging.error(f"翻译过程出错: {str(e)}")
            return ""

def main():
    # 加载配置
    config = TransformerConfig()
    config.device = 'cpu'  # 直接使用CPU
    
    logging.info("加载数据...")
    # 加载数据模块
    data = DataModule(
        batch_size=1,
        max_length=config.max_seq_length,
        config=config
    )
    data.setup()
    
    # 检查词表
    logging.info(f"源语言词表大小: {len(data.src_vocab)}")
    logging.info(f"目标语言词表大小: {len(data.tgt_vocab)}")
    
    # 检查特殊token
    logging.info(f"<unk> ID: {data.tgt_vocab['<unk>']}")
    logging.info(f"<pad> ID: {data.tgt_vocab['<pad>']}")
    logging.info(f"<bos> ID: {data.tgt_vocab['<bos>']}")
    logging.info(f"<eos> ID: {data.tgt_vocab['<eos>']}")
    
    logging.info("初始化模型...")
    
    # 查找最佳checkpoint - 优先选择主训练的模型
    checkpoint_files = []
    if os.path.exists(config.checkpoint_dir):
        all_files = os.listdir(config.checkpoint_dir)
        # 排除消融实验的checkpoint
        checkpoint_files = [f for f in all_files 
                           if f.startswith('best_model') 
                           and 'epoch' in f 
                           and not any(x in f for x in ['heads', 'layers', 'd_model', 'no_'])]
        
        # 如果没有找到主训练的,再看所有best_model
        if not checkpoint_files:
            checkpoint_files = [f for f in all_files if f.startswith('best_model')]
    
    if not checkpoint_files:
        logging.error("未找到checkpoint文件")
        logging.info("请先运行训练:")
        logging.info("  bash scripts/run.sh --mode train --device cuda")
        return
    
    # 按修改时间排序,选择最新的
    checkpoint_files.sort(key=lambda x: os.path.getmtime(
        os.path.join(config.checkpoint_dir, x)), reverse=True)
    
    checkpoint_path = os.path.join(config.checkpoint_dir, checkpoint_files[0])
    logging.info(f"找到 {len(checkpoint_files)} 个checkpoint文件")
    logging.info(f"使用最新的: {checkpoint_path}")
    
    try:
        # 先加载checkpoint以获取模型配置
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        
        # 从checkpoint的权重推断模型配置
        # 从embedding层推断d_model
        d_model = checkpoint['model_state_dict']['encoder_embedding.weight'].shape[1]
        # 从位置编码推断max_seq_length
        max_seq_length = checkpoint['model_state_dict']['positional_encoding.pe'].shape[1]
        # 计算层数
        num_layers = sum(1 for k in checkpoint['model_state_dict'].keys() if k.startswith('encoder.layers.') and k.endswith('.norm1.weight'))
        # 从第一层的注意力推断num_heads (d_model / d_k)
        # 假设d_k = d_model / num_heads, 我们使用标准配置
        if d_model == 128:
            num_heads = 4
        elif d_model == 256:
            num_heads = 8
        else:
            num_heads = 8
        
        logging.info(f"从checkpoint检测到的模型配置:")
        logging.info(f"  d_model: {d_model}")
        logging.info(f"  max_seq_length: {max_seq_length}")
        logging.info(f"  num_layers: {num_layers}")
        logging.info(f"  num_heads: {num_heads}")
        
        # 使用检测到的配置创建模型
        model = Transformer(
            src_vocab_size=len(data.src_vocab),
            tgt_vocab_size=len(data.tgt_vocab),
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=config.d_ff,
            dropout=0.0,
            max_seq_length=max_seq_length
        ).to(config.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logging.info("模型加载成功")
        
    except Exception as e:
        logging.error(f"加载模型失败: {str(e)}")
        return
    
    # 创建翻译器
    translator = TranslationGenerator(
        model, 
        data.src_vocab, 
        data.tgt_vocab, 
        config.device
    )
    
    # 从验证集中随机选择一些样本
    import random
    test_samples = random.sample(range(len(data.val_data)), min(20, len(data.val_data)))
    
    # 生成翻译样例
    results_file = 'results/translation_samples.md'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    logging.info("生成翻译样例...")
    
    # 添加统计信息
    total_samples = 0
    successful_samples = 0
    failed_samples = 0
    unk_heavy_samples = 0
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("# Transformer 翻译样例\n\n")
        f.write(f"模型配置: d_model={config.d_model}, layers={config.num_layers}, heads={config.num_heads}\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n\n")
        f.write("---\n\n")
        
        for i, sample_idx in enumerate(test_samples, 1):
            try:
                total_samples += 1
                src_tensor, tgt_tensor = data.val_data[sample_idx]
                
                # 解码源文本
                src_ids = src_tensor.numpy()
                src_words = []
                for idx in src_ids:
                    if idx == data.src_vocab['<pad>']:
                        break
                    if idx in [data.src_vocab['<bos>'], data.src_vocab['<eos>']]:
                        continue
                    word = data.src_vocab.get_itos()[idx]
                    src_words.append(word)
                src_text = ' '.join(src_words)
                
                # 解码参考翻译
                tgt_ids = tgt_tensor.numpy()
                ref_words = []
                for idx in tgt_ids:
                    if idx == data.tgt_vocab['<pad>']:
                        break
                    if idx in [data.tgt_vocab['<bos>'], data.tgt_vocab['<eos>']]:
                        continue
                    word = data.tgt_vocab.get_itos()[idx]
                    ref_words.append(word)
                ref_text = ''.join(ref_words)
                
                # 生成翻译
                try:
                    logging.info(f"\n生成样例 {i}...")
                    generated_text = translator.translate(src_text, data.src_tokenizer, temperature=0.8)
                    
                    if not generated_text or generated_text == "":
                        failed_samples += 1
                    elif generated_text.count('<unk>') / max(len(generated_text), 1) > 0.5:
                        unk_heavy_samples += 1
                    else:
                        successful_samples += 1
                        
                except Exception as e:
                    logging.error(f"样例 {i} 翻译失败: {str(e)}")
                    generated_text = "[翻译失败]"
                    failed_samples += 1
                
                # 写入文件
                f.write(f"## 样例 {i}\n\n")
                f.write(f"**源文本 (英文)**:\n> {src_text}\n\n")
                f.write(f"**参考翻译 (中文)**:\n> {ref_text}\n\n")
                f.write(f"**模型翻译 (中文)**:\n> {generated_text}\n\n")
                
                # 简单的质量评估
                if generated_text == ref_text:
                    f.write("✅ **完全匹配**\n\n")
                elif len(generated_text) > 0:
                    f.write("⚠️ **部分正确**\n\n")
                else:
                    f.write("❌ **翻译失败**\n\n")
                
                f.write("---\n\n")
                
                # 打印进度
                if i % 5 == 0:
                    logging.info(f"已生成 {i}/{len(test_samples)} 个样例")
            
            except Exception as e:
                logging.error(f"处理样例 {i} 时出错: {str(e)}")
                failed_samples += 1
                continue
        
        # 添加统计摘要
        f.write("\n## 统计摘要\n\n")
        f.write(f"- 总样例数: {total_samples}\n")
        f.write(f"- 成功生成: {successful_samples}\n")
        f.write(f"- 大量<unk>: {unk_heavy_samples}\n")
        f.write(f"- 生成失败: {failed_samples}\n")
    
    logging.info(f"翻译样例已保存到: {results_file}")
    logging.info(f"\n统计:")
    logging.info(f"  总样例: {total_samples}")
    logging.info(f"  成功: {successful_samples}")
    logging.info(f"  大量<unk>: {unk_heavy_samples}")
    logging.info(f"  失败: {failed_samples}")
    
    # 生成统计信息
    logging.info("\n生成统计摘要...")
    with open('results/translation_stats.txt', 'w', encoding='utf-8') as f:
        f.write("翻译样例统计\n")
        f.write("="*50 + "\n")
        f.write(f"总样例数: {len(test_samples)}\n")
        f.write(f"源语言词表大小: {len(data.src_vocab)}\n")
        f.write(f"目标语言词表大小: {len(data.tgt_vocab)}\n")
        f.write(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}\n")

if __name__ == '__main__':
    main()
