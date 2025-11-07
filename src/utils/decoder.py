import torch
import torch.nn.functional as F
from typing import List, Tuple
import logging

class BeamSearchDecoder:
    """Beam Search解码器"""
    
    def __init__(self, model, src_vocab, tgt_vocab, device='cuda', 
                 beam_size=5, max_length=128, length_penalty=0.6, 
                 repetition_penalty=1.2):
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        self.beam_size = beam_size
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty
        
        self.bos_id = tgt_vocab['<bos>']
        self.eos_id = tgt_vocab['<eos>']
        self.pad_id = tgt_vocab['<pad>']
    
    def _apply_length_penalty(self, scores, lengths):
        """应用长度惩罚
        
        使用Google NMT的长度惩罚公式:
        lp = ((5 + len) / 6) ^ alpha
        """
        penalty = ((5.0 + lengths) / 6.0) ** self.length_penalty
        return scores / penalty
    
    def _apply_repetition_penalty(self, logits, generated_ids):
        """应用重复惩罚
        
        降低已生成token的概率
        """
        for token_id in set(generated_ids):
            logits[token_id] = logits[token_id] / self.repetition_penalty
        return logits
    
    def beam_search(self, src, src_tokenizer):
        """Beam Search解码
        
        Args:
            src: 源句子文本
            src_tokenizer: 源语言分词器
            
        Returns:
            str: 最佳翻译结果
        """
        self.model.eval()
        
        with torch.no_grad():
            # 编码源句子
            src_tokens = src_tokenizer.tokenize(src)
            if len(src_tokens) > self.max_length - 2:
                src_tokens = src_tokens[:self.max_length - 2]
            
            src_ids = [self.src_vocab['<bos>']] + \
                     [self.src_vocab[t] for t in src_tokens] + \
                     [self.src_vocab['<eos>']]
            
            src_ids += [self.src_vocab['<pad>']] * (self.max_length - len(src_ids))
            src_ids = src_ids[:self.max_length]
            
            src_tensor = torch.tensor([src_ids], dtype=torch.long).to(self.device)
            
            # 初始化beam
            # 每个hypothesis: (score, tokens, finished)
            hypotheses = [(0.0, [self.bos_id], False)]
            
            for step in range(self.max_length - 1):
                all_candidates = []
                
                for score, tokens, finished in hypotheses:
                    if finished:
                        all_candidates.append((score, tokens, True))
                        continue
                    
                    # 构造目标序列
                    tgt_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
                    
                    # 前向传播
                    try:
                        output = self.model(src_tensor, tgt_tensor)
                        logits = output[0, -1, :]
                        
                        # 应用重复惩罚
                        logits = self._apply_repetition_penalty(logits, tokens)
                        
                        # 计算log概率
                        log_probs = F.log_softmax(logits, dim=-1)
                        
                        # 获取top-k候选
                        top_k_probs, top_k_indices = torch.topk(log_probs, self.beam_size * 2)
                        
                        # 为每个候选创建新hypothesis
                        for prob, idx in zip(top_k_probs, top_k_indices):
                            token_id = idx.item()
                            new_score = score + prob.item()
                            new_tokens = tokens + [token_id]
                            
                            # 检查是否结束
                            is_finished = (token_id == self.eos_id)
                            
                            all_candidates.append((new_score, new_tokens, is_finished))
                    
                    except Exception as e:
                        logging.warning(f"Beam search解码错误: {e}")
                        continue
                
                # 根据分数排序并选择top beam_size个hypothesis
                # 对于已完成的hypothesis,应用长度惩罚
                scored_candidates = []
                for score, tokens, finished in all_candidates:
                    if finished:
                        # 应用长度惩罚
                        penalized_score = self._apply_length_penalty(
                            score, len(tokens)
                        )
                    else:
                        penalized_score = score
                    scored_candidates.append((penalized_score, score, tokens, finished))
                
                scored_candidates.sort(reverse=True, key=lambda x: x[0])
                
                # 选择top beam_size
                hypotheses = [(score, tokens, finished) 
                             for _, score, tokens, finished in scored_candidates[:self.beam_size]]
                
                # 如果所有hypothesis都完成了,提前结束
                if all(finished for _, _, finished in hypotheses):
                    break
            
            # 选择最佳hypothesis
            best_hypothesis = max(hypotheses, key=lambda x: 
                                self._apply_length_penalty(x[0], len(x[1])))
            
            best_tokens = best_hypothesis[1]
            
            # 解码为文本
            words = []
            for token_id in best_tokens[1:]:  # 跳过<bos>
                if token_id == self.eos_id:
                    break
                try:
                    word = self.tgt_vocab.get_itos()[token_id]
                    words.append(word)
                except:
                    continue
            
            return ''.join(words)


class NucleusDecoder:
    """Nucleus (Top-p) Sampling解码器"""
    
    def __init__(self, model, src_vocab, tgt_vocab, device='cuda',
                 top_p=0.9, temperature=1.0, max_length=128):
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        self.top_p = top_p
        self.temperature = temperature
        self.max_length = max_length
        
        self.bos_id = tgt_vocab['<bos>']
        self.eos_id = tgt_vocab['<eos>']
        self.pad_id = tgt_vocab['<pad>']
    
    def nucleus_sampling(self, src, src_tokenizer):
        """Nucleus (Top-p) Sampling
        
        只从累积概率达到p的最小token集合中采样
        """
        self.model.eval()
        
        with torch.no_grad():
            # 编码源句子
            src_tokens = src_tokenizer.tokenize(src)
            if len(src_tokens) > self.max_length - 2:
                src_tokens = src_tokens[:self.max_length - 2]
            
            src_ids = [self.src_vocab['<bos>']] + \
                     [self.src_vocab[t] for t in src_tokens] + \
                     [self.src_vocab['<eos>']]
            
            src_ids += [self.src_vocab['<pad>']] * (self.max_length - len(src_ids))
            src_ids = src_ids[:self.max_length]
            
            src_tensor = torch.tensor([src_ids], dtype=torch.long).to(self.device)
            
            # 初始化生成序列
            generated = [self.bos_id]
            
            for step in range(self.max_length - 1):
                tgt_tensor = torch.tensor([generated], dtype=torch.long).to(self.device)
                
                try:
                    output = self.model(src_tensor, tgt_tensor)
                    logits = output[0, -1, :] / self.temperature
                    
                    # 计算概率
                    probs = F.softmax(logits, dim=-1)
                    
                    # 排序概率
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    
                    # 计算累积概率
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # 找到累积概率超过top_p的位置
                    cutoff_index = torch.where(cumulative_probs > self.top_p)[0]
                    if len(cutoff_index) > 0:
                        cutoff_index = cutoff_index[0].item() + 1
                    else:
                        cutoff_index = len(sorted_probs)
                    
                    # 只保留top-p的tokens
                    top_p_probs = sorted_probs[:cutoff_index]
                    top_p_indices = sorted_indices[:cutoff_index]
                    
                    # 重新归一化
                    top_p_probs = top_p_probs / top_p_probs.sum()
                    
                    # 采样
                    sampled_index = torch.multinomial(top_p_probs, 1)
                    next_token = top_p_indices[sampled_index].item()
                    
                    generated.append(next_token)
                    
                    if next_token == self.eos_id:
                        break
                
                except Exception as e:
                    logging.warning(f"Nucleus sampling解码错误: {e}")
                    break
            
            # 解码为文本
            words = []
            for token_id in generated[1:]:
                if token_id == self.eos_id:
                    break
                try:
                    word = self.tgt_vocab.get_itos()[token_id]
                    words.append(word)
                except:
                    continue
            
            return ''.join(words)


class EnsembleDecoder:
    """集成多个解码策略"""
    
    def __init__(self, model, src_vocab, tgt_vocab, device='cuda'):
        self.beam_decoder = BeamSearchDecoder(
            model, src_vocab, tgt_vocab, device,
            beam_size=5, length_penalty=0.6, repetition_penalty=1.2
        )
        
        self.nucleus_decoder = NucleusDecoder(
            model, src_vocab, tgt_vocab, device,
            top_p=0.9, temperature=0.8
        )
    
    def translate(self, src, src_tokenizer, strategy='beam'):
        """使用指定策略翻译
        
        Args:
            src: 源文本
            src_tokenizer: 分词器
            strategy: 'beam', 'nucleus', 或 'ensemble'
            
        Returns:
            str: 翻译结果
        """
        if strategy == 'beam':
            return self.beam_decoder.beam_search(src, src_tokenizer)
        elif strategy == 'nucleus':
            return self.nucleus_decoder.nucleus_sampling(src, src_tokenizer)
        elif strategy == 'ensemble':
            # 生成多个候选并选择最佳
            candidates = []
            candidates.append(self.beam_decoder.beam_search(src, src_tokenizer))
            candidates.append(self.nucleus_decoder.nucleus_sampling(src, src_tokenizer))
            
            # 简单策略:返回beam search结果(通常更可靠)
            return candidates[0]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
