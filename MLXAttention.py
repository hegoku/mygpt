import mlx.core as mx
import mlx.nn as nn
from typing import Tuple

class LlamaMultiHeadAttention(nn.Module):
    freqs_cis = None
    mask = None

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
        self.rope = nn.RoPE(self.d_q, traditional=True)
        # if LlamaMultiHeadAttention.freqs_cis is None:
            # LlamaMultiHeadAttention.freqs_cis = precompute_freqs_cis(self.d_q, max_context)

        if LlamaMultiHeadAttention.mask is None:
            LlamaMultiHeadAttention.mask = nn.MultiHeadAttention.create_additive_causal_mask(self.max_context)

    def __call__(self, embedding_word):
        token_num = embedding_word.shape[-2]
        q = self.w_Q(embedding_word)
        k = self.w_K(embedding_word)
        v = self.w_V(embedding_word)

        if LlamaMultiHeadAttention.mask.dtype!=q.dtype:
            LlamaMultiHeadAttention.mask = LlamaMultiHeadAttention.mask.astype(q.dtype)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # q = q.view(b, num_tokens, self.num_heads, self.head_dim)
        # k = k.view(b, num_tokens, self.num_heads, self.head_dim)
        # v = v.view(b, num_tokens, self.num_heads, self.head_dim)
        # 为了支持不是batch的，改用下方式
        # q = q.view(b, num_tokens, self.num_heads, self.head_dim) or # q = q.view(num_tokens, self.num_heads, self.head_dim)
        q = q.reshape(*q.shape[:-1], self.head_num, self.d_q)
        k = k.reshape(*k.shape[:-1], self.head_num, self.d_q)
        v = v.reshape(*v.shape[:-1], self.head_num, self.d_v)

        shape = []
        for i in range(len(q.shape)):
            shape.append(i)
        tmp = shape[-2]
        shape[-2] = shape[-3]
        shape[-3] = tmp

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        q = q.transpose(shape)
        k = k.transpose(shape)
        v = v.transpose(shape)

        q = self.rope(q.astype(mx.float32)).astype(q.dtype)
        k = self.rope(k.astype(mx.float32)).astype(k.dtype)
        # q, k = apply_rotary_emb(q, k, freqs_cis=LlamaMultiHeadAttention.freqs_cis[:token_num,:])

        shape = []
        for i in range(len(k.shape)):
            shape.append(i)
        tmp = shape[-2]
        shape[-2] = shape[-1]
        shape[-1] = tmp
        scores = q @ k.transpose(shape)
        scores = scores + LlamaMultiHeadAttention.mask[:token_num, :token_num]
        weights = mx.softmax(scores / self.softmax_factor, axis=-1)

        # Shape: (b, num_tokens, num_heads, head_dim)
        shape = []
        for i in range(len(v.shape)):
            shape.append(i)
        tmp = shape[-2]
        shape[-2] = shape[-3]
        shape[-3] = tmp
        output = (weights @ v).transpose(shape)

        # Combine heads, where self.embedding_size = self.head_num * self.d_q
        output = output.reshape(*output.shape[:-2], self.embedding_size)
        output = self.output(output)
        return output

def polar(a, b):
  return a * mx.exp(1j * b)

# 生成旋转矩阵
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度\theta_i
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2)[: (dim // 2)] / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = mx.arange(seq_len, dtype=mx.float32)
    # freqs.shape = [seq_len, dim // 2] 
    freqs = mx.outer(t, freqs)  # 计算m * \theta

    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = polar(mx.ones_like(freqs), freqs) 
    return freqs_cis

# 旋转位置编码计算
def apply_rotary_emb(
    xq: mx.array,
    xk: mx.array,
    freqs_cis: mx.array,
) -> Tuple[mx.array, mx.array]:
    # xq.shape = [batch_size, head_num, seq_len, dim]
    # xq_.shape = [batch_size, head_num, seq_len, dim // 2, 2]
    xq_ = xq.astype(mx.float32).reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.astype(mx.float32).reshape(*xk.shape[:-1], -1, 2)

    # 转为复数域
    xq_ = mx.view_as_complex(xq_)
    xk_ = mx.view_as_complex(xk_)

    # 应用旋转操作，然后将结果转回实数域
    # xq_out.shape = [batch_size, head_num, seq_len, dim // 2, 2] -> [batch_size, head_num, seq_len, dim]
    xq_out = mx.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = mx.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)