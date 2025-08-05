from transformers import PreTrainedModel, PretrainedConfig
import MyGPT
import torch.nn as nn
import torch.nn.functional as F
import DeepseekTokenizer
import torch

class MyGPTPretrainedModelConfig(PretrainedConfig):
    model_type = "mygpt2"  # 唯一标识名
    def __init__(self, vocab_size=50000, max_context=1024, embedding_dim=768, layers=12, head_num=12, dropout=0.0, pad_token_id=-100, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.pad_token_id = pad_token_id
        self.layers = layers
        self.head_num = head_num
        self.dropout = dropout
        self.max_context = max_context

class MyGPTPretrainedModel(PreTrainedModel):
    config_class = MyGPTPretrainedModelConfig  # 关联配置

    def __init__(self, config):
        super().__init__(config)
        self.embedding_dim = config.embedding_dim
        self.max_context = config.max_context
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_token_id)
        self.drop_emb = nn.Dropout(config.dropout)
        self.transformer_layers = nn.ModuleList([MyGPT.TransformerBlock(config.max_context, config.embedding_dim, config.head_num) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.embedding_dim, eps=1e-5)
        # self.output = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        self.post_init()

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, (nn.Linear)) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, labels=None, **kwargs):
        embedding = self.token_embedding(input_ids)
        embedding = self.drop_emb(embedding)
        for layer in self.transformer_layers:
            embedding = layer(embedding)
        embedding = self.norm(embedding)
        # embedding = self.output(embedding)
        embedding = F.linear(embedding, self.token_embedding.weight)
        
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(embedding.flatten(0, 1), labels.flatten(), ignore_index=self.config.pad_token_id)
        return {"loss": loss, "logits": embedding}

def generate_text(model, idx, max_tokens:int, max_context:int, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_tokens):
        idx_cond = idx[-max_context:]
        with torch.no_grad():
            logits = model(idx_cond)['logits']

        logits = logits[-1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[-1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=0)
    return idx