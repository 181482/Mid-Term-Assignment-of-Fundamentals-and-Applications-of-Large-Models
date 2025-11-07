import random
import torch

class DataAugmentation:
    """简单的数据增强技术"""
    
    def __init__(self, vocab, unk_id, pad_id, bos_id, eos_id):
        self.vocab = vocab
        self.unk_id = unk_id
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
    
    def word_dropout(self, tokens, drop_prob=0.1):
        """随机将一些词替换为<unk>"""
        mask = torch.rand(tokens.shape) < drop_prob
        # 不要dropout特殊token
        special_mask = (tokens == self.pad_id) | (tokens == self.bos_id) | (tokens == self.eos_id)
        mask = mask & (~special_mask)
        tokens = tokens.clone()
        tokens[mask] = self.unk_id
        return tokens
    
    def token_cutoff(self, tokens, cutoff_prob=0.1):
        """随机截断序列"""
        if random.random() < cutoff_prob:
            # 找到有效长度
            valid_len = (tokens != self.pad_id).sum().item()
            if valid_len > 5:  # 至少保留5个token
                cutoff_point = random.randint(5, valid_len)
                tokens = tokens.clone()
                tokens[cutoff_point:] = self.pad_id
        return tokens
