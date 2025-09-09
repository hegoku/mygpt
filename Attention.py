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
    freqs_cis = None
    mask = None
    cos = None
    sin = None

    def __init__(self, max_context:int, embedding_dim:int, head_num:int, dropout=0.0, bias=False):
        super().__init__()
        self.max_context = max_context
        self.embedding_size = embedding_dim
        self.head_num = head_num
        self.d_q = self.embedding_size // head_num
        self.d_v = self.d_q
        self.softmax_factor = self.d_q**0.5

        self.w_Q = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.w_K = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.w_V = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.output = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        # if MultiHeadAttention.freqs_cis is None:
            # MultiHeadAttention.freqs_cis = precompute_freqs_cis(self.d_q, max_context)

        if MultiHeadAttention.mask is None:
            MultiHeadAttention.mask = torch.triu(torch.ones(self.max_context, self.max_context), diagonal=1)

        if MultiHeadAttention.cos is None:
            MultiHeadAttention.cos, MultiHeadAttention.sin = precompute_rope_params(self.d_q, context_length=max_context)

        self.linear_bias = bias
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedding_word, padding_mask=None):
        token_num = embedding_word.shape[-2]
        q = self.w_Q(embedding_word)
        k = self.w_K(embedding_word)
        v = self.w_V(embedding_word)
        # if MultiHeadAttention.freqs_cis.device!=q.device:
            # MultiHeadAttention.freqs_cis = MultiHeadAttention.freqs_cis.to(q.device)
        if MultiHeadAttention.mask.device!=q.device:
            MultiHeadAttention.mask = MultiHeadAttention.mask.to(q.device)
            MultiHeadAttention.cos = MultiHeadAttention.cos.to(q.device)
            MultiHeadAttention.sin = MultiHeadAttention.sin.to(q.device)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # q = q.view(b, num_tokens, self.num_heads, self.head_dim)
        # k = k.view(b, num_tokens, self.num_heads, self.head_dim)
        # v = v.view(b, num_tokens, self.num_heads, self.head_dim)
        # 为了支持不是batch的，改用下方式
        # q = q.view(b, num_tokens, self.num_heads, self.head_dim) or # q = q.view(num_tokens, self.num_heads, self.head_dim)
        q = q.view(*q.shape[:-1], self.head_num, self.d_q)
        k = k.view(*k.shape[:-1], self.head_num, self.d_q)
        v = v.view(*v.shape[:-1], self.head_num, self.d_v)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        q = q.transpose(-3, -2)
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)

        # q, k = apply_rotary_emb(q, k, freqs_cis=MultiHeadAttention.freqs_cis[:token_num,:])

        q = compute_rope(q, MultiHeadAttention.cos, MultiHeadAttention.sin)
        k = compute_rope(k, MultiHeadAttention.cos, MultiHeadAttention.sin)

        mask = MultiHeadAttention.mask.bool()[:token_num, :token_num]
        if padding_mask is not None:
            mask = mask | padding_mask

        scores = q @ k.transpose(-2, -1)
        scores.masked_fill_(mask, -torch.inf)
        weights = torch.softmax(scores / self.softmax_factor, dim=-1)
        weights = self.dropout(weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        output = (weights @ v).transpose(-3, -2)

        # Combine heads, where self.embedding_size = self.head_num * self.d_q
        output = output.reshape(*output.shape[:-2], self.embedding_size)
        output = self.output(output)
        return output

class LlamaMultiHeadAttention(nn.Module):
    freqs_cis = None
    mask = None
    cos = None
    sin = None

    def __init__(self, max_context:int, embedding_dim:int, head_num:int, rope_theta:float=10000.0):
        super().__init__()
        assert embedding_dim % head_num == 0, "embedding_dim must be divisible by head_num"
        self.max_context = max_context
        self.embedding_size = embedding_dim
        self.head_num = head_num
        self.d_q = self.embedding_size // head_num
        self.d_v = self.d_q
        self.softmax_factor = self.d_q**0.5

        self.w_Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.w_K = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.w_V = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.output = nn.Linear(embedding_dim, embedding_dim, bias=False)
        # self.register_buffer("mask", torch.triu(torch.ones(self.max_context, self.max_context), diagonal=1))
        # if LlamaMultiHeadAttention.freqs_cis is None:
            # LlamaMultiHeadAttention.freqs_cis = precompute_freqs_cis(self.d_q, max_context, theta=rope_theta)

        if LlamaMultiHeadAttention.mask is None:
            LlamaMultiHeadAttention.mask = torch.triu(torch.ones(self.max_context, self.max_context), diagonal=1)

        if LlamaMultiHeadAttention.cos is None:
            LlamaMultiHeadAttention.cos, LlamaMultiHeadAttention.sin = precompute_rope_params(self.d_q, context_length=max_context, theta_base=rope_theta)

    def forward(self, embedding_word, padding_mask=None):
        token_num = embedding_word.shape[-2]
        q = self.w_Q(embedding_word)
        k = self.w_K(embedding_word)
        v = self.w_V(embedding_word)
        # if LlamaMultiHeadAttention.freqs_cis.device!=q.device:
            # LlamaMultiHeadAttention.freqs_cis = LlamaMultiHeadAttention.freqs_cis.to(q.device)
        if LlamaMultiHeadAttention.mask.device!=q.device:
            LlamaMultiHeadAttention.mask = LlamaMultiHeadAttention.mask.to(q.device)
            LlamaMultiHeadAttention.cos = LlamaMultiHeadAttention.cos.to(q.device)
            LlamaMultiHeadAttention.sin = LlamaMultiHeadAttention.sin.to(q.device)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # q = q.view(b, num_tokens, self.num_heads, self.head_dim)
        # k = k.view(b, num_tokens, self.num_heads, self.head_dim)
        # v = v.view(b, num_tokens, self.num_heads, self.head_dim)
        # 为了支持不是batch的，改用下方式
        # q = q.view(b, num_tokens, self.num_heads, self.head_dim) or # q = q.view(num_tokens, self.num_heads, self.head_dim)
        q = q.view(*q.shape[:-1], self.head_num, self.d_q)
        k = k.view(*k.shape[:-1], self.head_num, self.d_q)
        v = v.view(*v.shape[:-1], self.head_num, self.d_v)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        q = q.transpose(-3, -2)
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)

        # q, k = apply_rotary_emb(q, k, freqs_cis=LlamaMultiHeadAttention.freqs_cis[:token_num,:])
        q = compute_rope(q, LlamaMultiHeadAttention.cos, LlamaMultiHeadAttention.sin)
        k = compute_rope(k, LlamaMultiHeadAttention.cos, LlamaMultiHeadAttention.sin)

        mask = LlamaMultiHeadAttention.mask.bool()[:token_num, :token_num]
        if padding_mask is not None:
            mask = mask | padding_mask

        scores = q @ k.transpose(-2, -1)
        scores.masked_fill_(mask, -torch.inf)
        weights = torch.softmax(scores / self.softmax_factor, dim=-1)

        # Shape: (b, num_tokens, num_heads, head_dim)
        output = (weights @ v).transpose(-3, -2)

        # output = nn.functional.scaled_dot_product_attention(
            # q, k, v, attn_mask=None, dropout_p=0, is_causal=True)
        # output = output.transpose(-3, -2)

        # Combine heads, where self.embedding_size = self.head_num * self.d_q
        output = output.reshape(*output.shape[:-2], self.embedding_size)
        output = self.output(output)
        return output


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
    # xq.shape = [batch_size, head_num, seq_len, dim]
    # xq_.shape = [batch_size, head_num, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

    # 转为复数域
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)

    # 应用旋转操作，然后将结果转回实数域
    # xq_out.shape = [batch_size, head_num, seq_len, dim // 2, 2] -> [batch_size, head_num, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))

    # Generate position indices
    positions = torch.arange(context_length)

    # Compute the angles
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)
    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

def compute_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    # batch_size, num_heads, seq_len, head_dim = x.shape
    seq_len = x.shape[-2]
    head_dim = x.shape[-1]
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    if x.ndim==4:
        cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
        sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
    else:
        cos = cos[:seq_len, :].unsqueeze(0)  # Shape: (1, seq_len, head_dim)
        sin = sin[:seq_len, :].unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)