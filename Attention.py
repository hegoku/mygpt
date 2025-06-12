import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, embedding_word):
        token_num = embedding_word.shape[-2]
        q = self.w_Q(embedding_word)
        k = self.w_K(embedding_word)
        v = self.w_V(embedding_word)
        scores = q.matmul(k.transpose(embedding_word.ndim-2, embedding_word.ndim-1))
        masked = scores.masked_fill(self.mask.bool()[:token_num, :token_num], -torch.inf)
        weights = F.softmax(masked / self.d_q**0.5, dim=-1)
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

    def forward(self, embedding_words):
        output = torch.cat([head(embedding_words) for head in self.heads], dim=-1)
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