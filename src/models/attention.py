import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """计算缩放点积注意力
        
        Args:
            Q: shape [batch_size, num_heads, seq_len_q, d_k]
            K: shape [batch_size, num_heads, seq_len_k, d_k]
            V: shape [batch_size, num_heads, seq_len_v, d_k]
            mask: shape [batch_size, 1, seq_len_q, seq_len_k]
        """
        d_k = Q.size(-1)
        
        # 确保在正确的设备和数据类型上
        Q = Q.float()  # 临时转换为float32进行计算
        K = K.float()
        V = V.float()
        
        # 检查输入维度
        assert Q.size(-1) == K.size(-1) == V.size(-1), "Q, K, V 的特征维度必须相同"
        assert K.size(-2) == V.size(-2), "K 和 V 的序列长度必须相同"
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # [batch_size, num_heads, seq_len_q, seq_len_k]
        
        if mask is not None:
            mask = mask.to(Q.device)
            # 确保掩码维度正确
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # 添加head维度
            mask = mask.expand(-1, Q.size(1), -1, -1)
            
            # 使用较小的负值来避免溢出
            scores = scores.masked_fill(mask == 0, -1e4)
            
        # 使用更稳定的softmax计算方式
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 检查输入维度
        if len(Q.shape) != 3 or len(K.shape) != 3 or len(V.shape) != 3:
            raise ValueError(f"输入维度错误: Q:{Q.shape}, K:{K.shape}, V:{V.shape}")
        
        # 转换输入tensor为contiguous并设置正确的形状
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k)
        
        # 改变维度顺序并确保内存连续
        Q = Q.transpose(1, 2).contiguous()  # [batch, heads, seq_len, d_k]
        K = K.transpose(1, 2).contiguous()
        V = V.transpose(1, 2).contiguous()
        
        x, attention = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 恢复原始维度
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(x)
