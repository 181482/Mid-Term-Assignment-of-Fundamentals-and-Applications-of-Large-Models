import logging
from utils.data_loader import DataModule
from config import TransformerConfig
import random

logging.basicConfig(level=logging.INFO)

def inspect_raw_data():
    """检查原始数据和预处理后的数据"""
    config = TransformerConfig()
    config.device = 'cpu'
    
    logging.info("="*60)
    logging.info("开始加载数据...")
    logging.info("="*60)
    
    # 初始化数据模块
    data = DataModule(
        batch_size=1,
        max_length=128,
        config=config
    )
    
    # 加载数据
    data.setup()
    
    logging.info("\n" + "="*60)
    logging.info("数据集统计:")
    logging.info("="*60)
    logging.info(f"训练集大小: {len(data.train_data)}")
    logging.info(f"验证集大小: {len(data.val_data)}")
    logging.info(f"测试集大小: {len(data.test_data)}")
    logging.info(f"源语言词表大小: {len(data.src_vocab)}")
    logging.info(f"目标语言词表大小: {len(data.tgt_vocab)}")
    
    # 检查词表中的特殊标记
    logging.info("\n" + "="*60)
    logging.info("特殊标记ID:")
    logging.info("="*60)
    for special_token in ['<unk>', '<pad>', '<bos>', '<eos>']:
        src_id = data.src_vocab[special_token]
        tgt_id = data.tgt_vocab[special_token]
        logging.info(f"{special_token}: 源={src_id}, 目标={tgt_id}")
    
    # 检查词表中<unk>的频率
    logging.info("\n" + "="*60)
    logging.info("词表样本 (前20个tokens):")
    logging.info("="*60)
    src_itos = data.src_vocab.get_itos()
    tgt_itos = data.tgt_vocab.get_itos()
    logging.info(f"源语言: {src_itos[:20]}")
    logging.info(f"目标语言: {tgt_itos[:20]}")
    
    # 随机抽取样本检查
    logging.info("\n" + "="*60)
    logging.info("随机样本检查 (训练集):")
    logging.info("="*60)
    
    num_samples = 10
    sample_indices = random.sample(range(len(data.train_data)), min(num_samples, len(data.train_data)))
    
    for i, idx in enumerate(sample_indices, 1):
        src_tensor, tgt_tensor = data.train_data[idx]
        
        # 解码源文本
        src_ids = src_tensor.numpy()
        src_tokens = []
        for token_id in src_ids:
            if token_id == data.src_vocab['<pad>']:
                break
            if token_id in [data.src_vocab['<bos>'], data.src_vocab['<eos>']]:
                continue
            token = src_itos[token_id]
            src_tokens.append(token)
        
        # 解码目标文本
        tgt_ids = tgt_tensor.numpy()
        tgt_tokens = []
        unk_count = 0
        for token_id in tgt_ids:
            if token_id == data.tgt_vocab['<pad>']:
                break
            if token_id in [data.tgt_vocab['<bos>'], data.tgt_vocab['<eos>']]:
                continue
            token = tgt_itos[token_id]
            tgt_tokens.append(token)
            if token == '<unk>':
                unk_count += 1
        
        logging.info(f"\n--- 样本 {i} ---")
        logging.info(f"源文本 ({len(src_tokens)} tokens):")
        logging.info(f"  {' '.join(src_tokens[:50])}..." if len(src_tokens) > 50 else f"  {' '.join(src_tokens)}")
        logging.info(f"目标文本 ({len(tgt_tokens)} tokens, {unk_count} <unk>):")
        logging.info(f"  {''.join(tgt_tokens[:50])}..." if len(tgt_tokens) > 50 else f"  {''.join(tgt_tokens)}")
        logging.info(f"<unk>比例: {unk_count/max(len(tgt_tokens), 1)*100:.1f}%")
    
    # 统计<unk>的总体分布
    logging.info("\n" + "="*60)
    logging.info("全局<unk>统计 (验证集):")
    logging.info("="*60)
    
    total_tokens = 0
    total_unk = 0
    samples_with_high_unk = 0
    
    for src_tensor, tgt_tensor in data.val_data:
        tgt_ids = tgt_tensor.numpy()
        sample_tokens = 0
        sample_unk = 0
        
        for token_id in tgt_ids:
            if token_id == data.tgt_vocab['<pad>']:
                break
            if token_id in [data.tgt_vocab['<bos>'], data.tgt_vocab['<eos>']]:
                continue
            sample_tokens += 1
            total_tokens += 1
            if token_id == data.tgt_vocab['<unk>']:
                sample_unk += 1
                total_unk += 1
        
        if sample_tokens > 0 and sample_unk / sample_tokens > 0.5:
            samples_with_high_unk += 1
    
    logging.info(f"总token数: {total_tokens}")
    logging.info(f"<unk>数量: {total_unk}")
    logging.info(f"<unk>比例: {total_unk/max(total_tokens, 1)*100:.1f}%")
    logging.info(f"高<unk>样本数 (>50%): {samples_with_high_unk}/{len(data.val_data)}")
    
    # 保存详细报告
    output_file = 'results/data_inspection_report.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("数据检查报告\n")
        f.write("="*60 + "\n\n")
        
        f.write("数据集统计:\n")
        f.write(f"  训练集: {len(data.train_data)} 句对\n")
        f.write(f"  验证集: {len(data.val_data)} 句对\n")
        f.write(f"  测试集: {len(data.test_data)} 句对\n")
        f.write(f"  源语言词表: {len(data.src_vocab)} tokens\n")
        f.write(f"  目标语言词表: {len(data.tgt_vocab)} tokens\n\n")
        
        f.write("<unk>统计:\n")
        f.write(f"  总token数: {total_tokens}\n")
        f.write(f"  <unk>数量: {total_unk}\n")
        f.write(f"  <unk>比例: {total_unk/max(total_tokens, 1)*100:.1f}%\n")
        f.write(f"  高<unk>样本: {samples_with_high_unk}/{len(data.val_data)}\n\n")
        
        f.write("问题分析:\n")
        if total_unk / max(total_tokens, 1) > 0.3:
            f.write("  ❌ 数据中包含大量<unk>标记 (>30%)\n")
            f.write("  建议:\n")
            f.write("    1. 检查原始数据质量\n")
            f.write("    2. 使用更好的中文分词器 (如jieba)\n")
            f.write("    3. 降低词表的min_freq阈值\n")
            f.write("    4. 考虑使用BPE/SentencePiece分词\n")
        else:
            f.write("  ✓ <unk>比例在可接受范围内\n")
    
    logging.info(f"\n详细报告已保存到: {output_file}")

if __name__ == '__main__':
    inspect_raw_data()
