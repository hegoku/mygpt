import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SelfAttention(nn.Module):
    def __init__(self, max_context:int, embedding_dim:int, d_q:int, d_v:int, bias=False):
        super().__init__()
        self.max_context = max_context
        self.embedding_size = embedding_dim
        self.d_q = d_q
        self.d_v = d_v
        self.linear_bias = bias
        self.w_Q = nn.Linear(in_features=embedding_dim, out_features=d_q, bias=bias)
        self.w_K = nn.Linear(in_features=embedding_dim, out_features=d_q, bias=bias)
        self.w_V = nn.Linear(in_features=embedding_dim, out_features=d_v, bias=bias)
        self.mask = torch.triu(torch.ones(self.max_context, self.max_context), diagonal=1)

    def forward(self, embedding_word):
        q = self.w_Q(embedding_word)
        k = self.w_K(embedding_word)
        v = self.w_V(embedding_word)
        scores = q.matmul(k.T)
        masked = scores.masked_fill(self.mask.bool(), -torch.inf)
        weights = F.softmax(masked / self.d_q**0.5, dim=-1)
        output = weights.matmul(v)
        return output

class CausalAttention(nn.Module):
    def __init__(self, max_context:int, embedding_dim:int, d_q:int, d_v:int, dropout=0.0, bias=False):
        super().__init__()
        self.max_context = max_context
        self.embedding_size = embedding_dim
        self.d_q = d_q
        self.d_v = d_v
        self.linear_bias = bias
        self.dropout = dropout
        self.w_Q = nn.Linear(in_features=embedding_dim, out_features=d_q, bias=bias)
        self.w_K = nn.Linear(in_features=embedding_dim, out_features=d_q, bias=bias)
        self.w_V = nn.Linear(in_features=embedding_dim, out_features=d_v, bias=bias)
        self.register_buffer("mask", torch.triu(torch.ones(self.max_context, self.max_context), diagonal=1))
        self.dropout = nn.Dropout(dropout)
        self.freqs_cis = precompute_freqs_cis(d_q, max_context)
        self.softmax_factor = self.d_q**0.5

    def forward(self, embedding_word, pad_mask=None):
        token_num = embedding_word.shape[-2]
        q = self.w_Q(embedding_word)
        k = self.w_K(embedding_word)
        v = self.w_V(embedding_word)
        if self.freqs_cis.device!=q.device:
            self.freqs_cis = self.freqs_cis.to(q.device)
        q, k = apply_rotary_emb(q, k, freqs_cis=self.freqs_cis[:token_num,:])
        scores = q.matmul(k.transpose(embedding_word.ndim-2, embedding_word.ndim-1))
        masked = scores.masked_fill(self.mask.bool()[:token_num, :token_num], -torch.inf)
        weights = F.softmax(masked / self.softmax_factor, dim=-1)
        if pad_mask!=None:
            weights = weights.masked_fill(pad_mask, 0)
        weights = self.dropout(weights)
        output = weights.matmul(v)
        return output

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, max_context:int, embedding_dim:int, d_q:int, d_v:int, head_num:int, dropout=0.0, bias=False):
        super().__init__()
        self.max_context = max_context
        self.embedding_size = embedding_dim
        self.d_q = d_q
        self.d_v = d_v
        self.linear_bias = bias
        self.dropout = dropout
        self.head_num = head_num
        self.heads = nn.ModuleList([CausalAttention(max_context, embedding_dim, d_q, d_v, dropout, bias) for _ in range(head_num)])
        self.output = nn.Linear(in_features=d_v*head_num, out_features=embedding_dim, bias=bias)

    def forward(self, embedding_words, pad_mask=None):
        output = torch.cat([head(embedding_words, pad_mask) for head in self.heads], dim=-1)
        return self.output(output)

class MultiHeadAttention(nn.Module):
    def __init__(self, max_context:int, embedding_dim:int, d_q:int, d_v:int, head_num:int, dropout=0.0, bias=False):
        super().__init__()
        self.max_context = max_context
        self.embedding_size = embedding_dim
        self.d_q = d_q
        self.d_v = d_v
        self.linear_bias = bias
        self.dropout = dropout
        self.head_num = head_num

# 生成旋转矩阵
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度\theta_i
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, dtype=torch.float32)
    # freqs.shape = [seq_len, dim // 2] 
    freqs = torch.outer(t, freqs).float()  # 计算m * \theta

    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
    return freqs_cis

# 旋转位置编码计算
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

    # 转为复数域
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)

    # 应用旋转操作，然后将结果转回实数域
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)