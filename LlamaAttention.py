import torch
import torch.nn as nn
from typing import Tuple

class LlamaMultiHeadAttention(nn.Module):
    def __init__(self, max_context:int, embedding_dim:int, head_num:int):
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
        self.register_buffer("mask", torch.triu(torch.ones(self.max_context, self.max_context), diagonal=1))
        self.freqs_cis = precompute_freqs_cis(self.d_q, max_context)

    def forward(self, embedding_word):
        token_num = embedding_word.shape[-2]
        q = self.w_Q(embedding_word)
        k = self.w_K(embedding_word)
        v = self.w_V(embedding_word)
        if self.freqs_cis.device!=q.device:
            self.freqs_cis = self.freqs_cis.to(q.device)

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

        q, k = apply_rotary_emb(q, k, freqs_cis=self.freqs_cis[:token_num,:])

        scores = q @ k.transpose(-2, -1)
        scores.masked_fill_(self.mask.bool()[:token_num, :token_num], -torch.inf)
        weights = torch.softmax(scores / self.softmax_factor, dim=-1)

        # Shape: (b, num_tokens, num_heads, head_dim)
        output = (weights @ v).transpose(-3, -2)

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