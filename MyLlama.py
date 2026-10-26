import torch
import torch.nn as nn
import torch.nn.functional as F
import Attention
import math

class MyLlama(nn.Module):
    def __init__(self, tokenizer, layer:int, max_context:int, embedding_dim:int, head_num:int, ff_dim:int, rope_theta:float=10000.0):
        super().__init__()
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.max_context = max_context
        self.head_num = head_num
        self.token_embedding = nn.Embedding(tokenizer.len(), embedding_dim, padding_idx=tokenizer.pad_token_id)
        # self.transformer_layers = nn.Sequential(*[TransformerBlock(max_context, embedding_dim, head_num, ff_dim) for _ in range(layer)])
        self.transformer_layers = nn.ModuleList([TransformerBlock(max_context, embedding_dim, head_num, ff_dim, rope_theta) for _ in range(layer)])
        self.norm = nn.RMSNorm(embedding_dim, eps=1e-6)
        self.output = nn.Linear(embedding_dim, tokenizer.len(), bias=False)
        
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         fan_in = m.in_features
        #         torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
        #         if m.bias is not None:
        #             torch.nn.init.zeros_(m.bias)
        #     elif isinstance(m, nn.Embedding):
        #         torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
        #         m.weight.data[tokenizer.pad_token_id].zero_()

        # self.output.weight = self.token_embedding.weight

    def forward(self, token_ids, padding_mask=None):
        embedding = self.token_embedding(token_ids)
        for layer in self.transformer_layers:
            embedding = layer(embedding, padding_mask)
        # embedding = self.transformer_layers(embedding)
        embedding = self.norm(embedding)
        embedding = self.output(embedding)
        return embedding

class TransformerBlock(nn.Module):
    def __init__(self, max_context:int, embedding_dim:int, head_num:int, ff_dim:int, rope_theta:float=10000.0):
        super().__init__()
        self.head_num = head_num
        self.norm1 = nn.RMSNorm(embedding_dim, eps=1e-6)
        self.norm2 = nn.RMSNorm(embedding_dim, eps=1e-6)
        self.atten = Attention.LlamaMultiHeadAttention(max_context, embedding_dim, head_num, rope_theta=rope_theta)
        # self.ff = FeedForward(embedding_dim, 4*embedding_dim)
        self.ff = FeedForward(embedding_dim, ff_dim)

    def forward(self, x, padding_mask=None):
        shortcut = x
        x = self.norm1(x)
        x = self.atten(x, padding_mask)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut
        return x
    
class FeedForward(nn.Module):
    def __init__(self,embedding_dim: int, d_ff:int):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, d_ff, bias=False)
        self.fc2 = nn.Linear(embedding_dim, d_ff, bias=False)
        self.fc3 = nn.Linear(d_ff, embedding_dim, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
        return self.fc3(x)
    
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
