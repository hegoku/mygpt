import torch
import torch.nn as nn
import torch.nn.functional as F
import LlamaAttention

class MyLlama(nn.Module):
    def __init__(self, tokenizer, layer:int, max_context:int, embedding_dim:int, head_num:int):
        super().__init__()
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.max_context = max_context
        self.token_embedding = nn.Embedding(tokenizer.len(), embedding_dim)
        self.transformer_layers = nn.Sequential(*[TransformerBlock(max_context, embedding_dim, head_num) for _ in range(layer)])
        self.norm = nn.RMSNorm(embedding_dim)
        self.output = nn.Linear(embedding_dim, tokenizer.len(), bias=False)

    def forward(self, token_ids):
        embedding = self.token_embedding(token_ids)
        embedding = self.transformer_layers(embedding)
        embedding = self.norm(embedding)
        embedding = self.output(embedding)
        return embedding

class TransformerBlock(nn.Module):
    def __init__(self, max_context:int, embedding_dim:int, head_num:int):
        super().__init__()
        self.norm1 = nn.RMSNorm(embedding_dim)
        self.norm2 = nn.RMSNorm(embedding_dim)
        self.atten = LlamaAttention.LlamaMultiHeadAttention(max_context, embedding_dim, head_num)
        self.ff = FeedForward(embedding_dim, 4*embedding_dim)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.atten(x)
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
