import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .positional_encoding import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model=256,  # 改为256与config一致
                 num_heads=8,  # 改为8与config一致
                 num_layers=4,  # 改为4与config一致
                 d_ff=1024,  # 改为1024与config一致
                 max_seq_length=128,  # 改为128与config一致
                 dropout=0.1,
                 use_positional_encoding=True):
        super().__init__()
        
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.use_positional_encoding = use_positional_encoding
        
        # 只在需要时创建位置编码
        if self.use_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)
        
        self.final_layer = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model

    def generate_square_subsequent_mask(self, sz):
        """生成用于解码器的后续掩码"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_padding_mask(self, seq):
        """生成用于填充的掩码"""
        # seq的形状: [batch_size, seq_len]
        mask = (seq != self.vocab['<pad>']).unsqueeze(1).unsqueeze(2)
        return mask  # 形状: [batch_size, 1, 1, seq_len]
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 生成掩码
        if src_mask is None:
            # [batch_size, 1, 1, src_len]
            src_mask = torch.ones(src.shape[0], 1, 1, src.shape[1]).to(src.device)
        
        if tgt_mask is None:
            # [tgt_len, tgt_len]
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            # 扩展为4D张量 [batch_size, 1, tgt_len, tgt_len]
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0).expand(tgt.size(0), -1, -1, -1)
        
        # 添加嵌入和位置编码
        src = self.encoder_embedding(src)
        if self.use_positional_encoding:
            src = self.positional_encoding(src)
        
        tgt = self.decoder_embedding(tgt)
        if self.use_positional_encoding:
            tgt = self.positional_encoding(tgt)
        
        # 确保输入维度正确
        assert src.size(-1) == self.d_model, f"编码器输入维度错误: {src.size(-1)} != {self.d_model}"
        assert tgt.size(-1) == self.d_model, f"解码器输入维度错误: {tgt.size(-1)} != {self.d_model}"
        
        # 前向传播
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        
        output = self.final_layer(dec_output)
        return output

