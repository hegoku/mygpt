import mlx.core as mx
import mlx.nn as nn
import MLXAttention


class MLXLlama(nn.Module):
    def __init__(self, tokenizer, layer:int, max_context:int, embedding_dim:int, head_num:int):
        super().__init__()
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.max_context = max_context
        self.head_num = head_num
        self.token_embedding = nn.Embedding(tokenizer.len(), embedding_dim)
        self.transformer_layers = nn.Sequential(*[TransformerBlock(max_context, embedding_dim, head_num) for _ in range(layer)])
        self.norm = nn.RMSNorm(embedding_dim, eps=1e-08)
        self.output = nn.Linear(embedding_dim, tokenizer.len(), bias=False)

    def __call__(self, token_ids):
        embedding = self.token_embedding(token_ids)
        embedding = self.transformer_layers(embedding)
        embedding = self.norm(embedding)
        embedding = self.output(embedding)
        return embedding

class TransformerBlock(nn.Module):
    def __init__(self, max_context:int, embedding_dim:int, head_num:int):
        super().__init__()
        self.head_num = head_num
        self.norm1 = nn.RMSNorm(embedding_dim, eps=1e-08)
        self.norm2 = nn.RMSNorm(embedding_dim, eps=1e-08)
        self.atten = MLXAttention.LlamaMultiHeadAttention(max_context, embedding_dim, head_num)
        self.ff = FeedForward(embedding_dim, int((8/3)*embedding_dim))

    def __call__(self, x):
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

    def __call__(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
        return self.fc3(x)
    
def generate_text_batch(model, idx, max_tokens:int, max_context:int):
    for _ in range(max_tokens):
        idx_cond = idx[:, -max_context:]
        logits = model(idx_cond)

        logits = logits[:,-1, :]
        probas = mx.softmax(logits, axis=-1)
        idx_next = mx.argmax(probas, axis=-1, keepdims=True)
        idx = mx.concatenate((idx, idx_next), axis=1)
    return idx

def generate_text(model, idx, max_tokens:int, max_context:int):
    for _ in range(max_tokens):
        idx_cond = idx[-max_context:]
        logits = model(idx_cond)

        logits = logits[-1, :]
        probas = mx.softmax(logits, axis=-1)
        idx_next = mx.argmax(probas, axis=-1, keepdims=True)
        idx = mx.concatenate((idx, idx_next), axis=1)
    return idx
