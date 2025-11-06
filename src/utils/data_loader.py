import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
import sacremoses
import jieba  # 使用jieba替代sacremoses处理中文
import html
import re
import logging
import os
import pickle
import xml.etree.ElementTree as ET

class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab=None, tgt_vocab=None):
        """
        Args:
            data: list of (src_tensor, tgt_tensor) tuples
            src_vocab: 源语言词表
            tgt_vocab: 目标语言词表
        """
        self.data = data
        self.src_vocab = src_vocab  # 添加词表引用
        self.tgt_vocab = tgt_vocab  # 添加词表引用
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

class DataModule:
    def __init__(self, batch_size=32, max_length=128, config=None):
        self.batch_size = batch_size
        self.max_length = max_length
        self.config = config
        
        # 初始化分词器 - 使用jieba处理中文
        self.src_tokenizer = sacremoses.MosesTokenizer(lang='en')  # 英文使用Moses
        self.tgt_tokenizer = lambda x: list(jieba.cut(x))  # 中文使用jieba
        
        # 数据路径
        self.data_dir = 'datasets/zh-en'
        
        # 缓存文件路径 - 更新缓存文件名以反映新的分词器
        self.cache_dir = os.path.join(self.data_dir, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, f'processed_data_jieba_len{max_length}.pkl')
        
        # 训练集: train.tags.zh-en.*
        self.train_src_file = 'train.tags.zh-en.en'
        self.train_tgt_file = 'train.tags.zh-en.zh'
        
        # 验证集: dev2010
        self.valid_src_file = 'IWSLT17.TED.dev2010.zh-en.en.xml'
        self.valid_tgt_file = 'IWSLT17.TED.dev2010.zh-en.zh.xml'
        
        # 测试集: tst2010-2012 (用于开发调试)
        # 最终测试集: tst2013-2015 (用于最终评估)
        self.test_years = ['2010', '2011', '2012']
        self.final_test_years = ['2013', '2014', '2015']
    
    def load_train_data(self, file_path):
        """
        加载训练集数据
        训练集格式: 包含XML标签行和文本行
        只保留不以'<'开头的文本行
        """
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('<'):
                    texts.append(html.unescape(line))
        return texts
    
    def load_xml_data(self, xml_file):
        """
        加载XML格式的数据 (dev和test集)
        提取<seg>标签中的文本
        """
        texts = []
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for seg in root.findall('.//seg'):
            if seg.text:
                texts.append(html.unescape(seg.text.strip()))
        return texts
    
    def load_test_data(self, years):
        """
        加载测试集数据
        可以加载多个年份的数据并合并
        """
        src_texts = []
        tgt_texts = []
        
        for year in years:
            src_file = f'IWSLT17.TED.tst{year}.zh-en.en.xml'
            tgt_file = f'IWSLT17.TED.tst{year}.zh-en.zh.xml'
            
            src_texts.extend(self.load_xml_data(
                os.path.join(self.data_dir, src_file)
            ))
            tgt_texts.extend(self.load_xml_data(
                os.path.join(self.data_dir, tgt_file)
            ))
        
        return src_texts, tgt_texts
    
    def _save_cache(self):
        """保存处理后的数据到缓存文件"""
        cache_data = {
            'src_vocab': self.src_vocab,
            'tgt_vocab': self.tgt_vocab,
            'train_data': self.train_data,
            'val_data': self.val_data,
            'test_data': self.test_data
        }
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        logging.info(f"数据已缓存到: {self.cache_file}")
    
    def _load_cache(self):
        """从缓存文件加载处理后的数据"""
        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.src_vocab = cache_data['src_vocab']
            self.tgt_vocab = cache_data['tgt_vocab']
            self.train_data = cache_data['train_data']
            self.val_data = cache_data['val_data']
            self.test_data = cache_data['test_data']
            
            logging.info("="*50)
            logging.info("从缓存加载数据集:")
            logging.info(f"训练集: {len(self.train_data)} 句对")
            logging.info(f"验证集: {len(self.val_data)} 句对")
            logging.info(f"测试集: {len(self.test_data)} 句对")
            logging.info(f"源语言词表: {len(self.src_vocab)}")
            logging.info(f"目标语言词表: {len(self.tgt_vocab)}")
            logging.info("="*50)
            
            return True
        except Exception as e:
            logging.info(f"无法加载缓存: {str(e)}")
            return False
    
    def setup(self):
        """加载和预处理数据集"""
        # 尝试从缓存加载
        if os.path.exists(self.cache_file):
            logging.info("检测到缓存文件，正在加载...")
            if self._load_cache():
                return
        
        # 如果没有缓存或加载失败，则重新处理数据
        try:
            logging.info("开始加载IWSLT17数据集...")
            
            # 1. 加载训练集 (train.tags)
            logging.info("加载训练集...")
            train_src = self.load_train_data(
                os.path.join(self.data_dir, self.train_src_file)
            )
            train_tgt = self.load_train_data(
                os.path.join(self.data_dir, self.train_tgt_file)
            )
            
            # 2. 加载验证集 (dev2010)
            logging.info("加载验证集 (dev2010)...")
            valid_src = self.load_xml_data(
                os.path.join(self.data_dir, self.valid_src_file)
            )
            valid_tgt = self.load_xml_data(
                os.path.join(self.data_dir, self.valid_tgt_file)
            )
            
            # 3. 加载测试集 (tst2010-2012，用于开发期间的评估)
            logging.info("加载测试集 (tst2010-2012)...")
            test_src, test_tgt = self.load_test_data(self.test_years)
            
            # 验证数据对齐
            assert len(train_src) == len(train_tgt), f"训练集不对齐: {len(train_src)} vs {len(train_tgt)}"
            assert len(valid_src) == len(valid_tgt), f"验证集不对齐: {len(valid_src)} vs {len(valid_tgt)}"
            assert len(test_src) == len(test_tgt), f"测试集不对齐: {len(test_src)} vs {len(test_tgt)}"
            
            # 对文本进行分词
            logging.info("分词处理...")
            train_src_tokens = [self.src_tokenizer.tokenize(text) for text in train_src]
            train_tgt_tokens = [self.tgt_tokenizer(text) for text in train_tgt]
            
            # 构建词表
            logging.info("构建词表...")
            self.src_vocab = build_vocab_from_iterator(
                train_src_tokens,
                min_freq=2,
                specials=['<unk>', '<pad>', '<bos>', '<eos>']
            )
            self.src_vocab.set_default_index(self.src_vocab['<unk>'])
            
            self.tgt_vocab = build_vocab_from_iterator(
                train_tgt_tokens,
                min_freq=2,
                specials=['<unk>', '<pad>', '<bos>', '<eos>']
            )
            self.tgt_vocab.set_default_index(self.tgt_vocab['<unk>'])
            
            # 处理数据集
            logging.info("处理数据集...")
            self.train_data = self._process_data(train_src, train_tgt)
            self.val_data = self._process_data(valid_src, valid_tgt)
            self.test_data = self._process_data(test_src, test_tgt)
            
            # 保存到缓存
            self._save_cache()
            
            # 打印统计信息
            logging.info("="*50)
            logging.info("IWSLT17数据集加载完成:")
            logging.info(f"训练集: {len(train_src)} 句对 -> {len(self.train_data)} 处理后")
            logging.info(f"验证集: {len(valid_src)} 句对 -> {len(self.val_data)} 处理后")
            logging.info(f"测试集: {len(test_src)} 句对 -> {len(self.test_data)} 处理后")
            logging.info(f"源语言词表: {len(self.src_vocab)}")
            logging.info(f"目标语言词表: {len(self.tgt_vocab)}")
            logging.info("="*50)
            
        except Exception as e:
            logging.error(f"数据集加载失败: {str(e)}")
            raise
    
    def _process_data(self, src_texts, tgt_texts):
        """将文本转换为token id序列"""
        processed = []
        skipped = 0
        
        for src, tgt in zip(src_texts, tgt_texts):
            try:
                # 分词
                src_tokens = self.src_tokenizer.tokenize(src)
                tgt_tokens = self.tgt_tokenizer(tgt)
                
                # 过滤过长的句子 - 基于token数量而非空格分割
                if len(src_tokens) > self.max_length - 2 or len(tgt_tokens) > self.max_length - 2:
                    skipped += 1
                    continue
                
                # 过滤过短的句子
                if len(src_tokens) < 1 or len(tgt_tokens) < 1:
                    skipped += 1
                    continue
                
                # 转换为ID并添加特殊token
                src_ids = [self.src_vocab['<bos>']] + \
                         [self.src_vocab[t] for t in src_tokens] + \
                         [self.src_vocab['<eos>']]
                         
                tgt_ids = [self.tgt_vocab['<bos>']] + \
                         [self.tgt_vocab[t] for t in tgt_tokens] + \
                         [self.tgt_vocab['<eos>']]
                
                # 填充到固定长度
                src_ids += [self.src_vocab['<pad>']] * (self.max_length - len(src_ids))
                tgt_ids += [self.tgt_vocab['<pad>']] * (self.max_length - len(tgt_ids))
                
                # 转换为tensor
                processed.append((
                    torch.tensor(src_ids[:self.max_length], dtype=torch.long),
                    torch.tensor(tgt_ids[:self.max_length], dtype=torch.long)
                ))
                
            except Exception as e:
                logging.warning(f"处理句对时出错: {str(e)}")
                skipped += 1
                continue
        
        if skipped > 0:
            logging.info(f"跳过了 {skipped} 个句对（过长、过短或处理失败）")
        
        return processed
    
    def get_dataloaders(self):
        """返回训练集、验证集和测试集的数据加载器"""
        train_dataset = TranslationDataset(
            self.train_data,
            src_vocab=self.src_vocab,
            tgt_vocab=self.tgt_vocab
        )
        val_dataset = TranslationDataset(
            self.val_data,
            src_vocab=self.src_vocab,
            tgt_vocab=self.tgt_vocab
        )
        test_dataset = TranslationDataset(
            self.test_data,
            src_vocab=self.src_vocab,
            tgt_vocab=self.tgt_vocab
        )
        
        return (
            DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                prefetch_factor=2,
                drop_last=True
            ),
            DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                num_workers=2,
                pin_memory=True
            ),
            DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                num_workers=2,
                pin_memory=True
            )
        )
