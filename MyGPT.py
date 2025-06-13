import torch
import torch.nn as nn
import torch.nn.functional as F
import Attention

class MyGPT(nn.Module):
    def __init__(self, tokenizer, layer:int, max_context:int, embedding_dim:int, d_q:int, d_v:int, head_num:int, dropout=0.0):
        super().__init__()
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.max_context = max_context
        self.token_embedding = nn.Embedding(tokenizer.len(), embedding_dim)
        self.drop_emb = nn.Dropout(dropout)
        self.transformer_layers = nn.Sequential(*[TransformerBlock(max_context, embedding_dim, d_q, d_v, head_num, dropout) for _ in range(layer)])
        self.norm = nn.LayerNorm(embedding_dim)
        self.output = nn.Linear(embedding_dim, tokenizer.len())
        self.p_embedding = nn.Embedding(self.max_context, self.embedding_dim)

    def forward(self, token_ids):
        embedding = self.token_embedding(token_ids)
        # embedding += self.p_embedding(torch.arange(token_ids.shape[-1], device=token_ids.device))
        embedding += self.positionEmbedding(token_ids.shape[-1], self.embedding_dim, device=token_ids.device, dtype=embedding.dtype)
        embedding = self.drop_emb(embedding)
        embedding = self.transformer_layers(embedding)
        embedding = self.norm(embedding)
        embedding = self.output(embedding)
        return embedding
    
    def positionEmbedding(self, seq_len, d_model, device, dtype, base=10000):
        res = torch.zeros(seq_len, d_model, device=device, dtype=dtype)
        exp_1 = torch.arange(d_model // 2, dtype=dtype, device=device)  # 初始化一半维度，sin位置编码的维度被分为了两部分
        exp_value = exp_1 / (d_model / 2)

        alpha = 1 / (base ** exp_value)  # size(dmodel/2)
        out = torch.arange(seq_len, dtype=dtype, device=device)[:, None] @ alpha[None, :]  # size(max_sequence_length, d_model/2)
        embedding_sin = torch.sin(out)
        embedding_cos = torch.cos(out)

        res[:, 0::2] = embedding_sin  # 奇数位置设置为sin
        res[:, 1::2] = embedding_cos  # 偶数位置设置为cos
        return res

class TransformerBlock(nn.Module):
    def __init__(self, max_context:int, embedding_dim:int, d_q:int, d_v:int, head_num:int, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.atten = Attention.MultiHeadAttentionWrapper(max_context, embedding_dim, d_q, d_v, head_num, dropout)
        self.drop_shortcut = nn.Dropout(dropout)
        self.ff = FeedForward(embedding_dim, 4*embedding_dim)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.atten(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
    
class FeedForward(nn.Module):
    def __init__(self,embedding_dim: int, d_ff:int):
        super().__init__()
        self.net = nn.Sequential( # multiplication of 4 comes from the fact that the dimensionality of input is x, but the inner layer dimensionality is 4*x
            nn.Linear(embedding_dim, d_ff), # linear layer with n_embd input and n_embd output
            nn.GELU(),# activation function, allows for non linearity (we use ReLU to get over vanishing gradients) -> vanishing gradients is essentially when
            nn.Linear(d_ff, embedding_dim),    #  the gradients are propagated backward from the output layer to the input layer, they can become very small (vanish) as they pass through many layers.
            # nn.Dropout(dropout)          # When the gradients become extremely small, the weights of the early layers are updated only by tiny amounts, if at all.
        )

    def forward(self, x):
        return self.net(x)
    
def generate_text(model, idx, max_tokens:int, max_context:int):
    for _ in range(max_tokens):
        idx_cond = idx[:, -max_context:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:,-1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
