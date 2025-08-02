from transformers import PreTrainedModel, PretrainedConfig
import MyGPT
import torch.nn as nn
import DeepseekTokenizer
import torch

class MyGPTPretrainedModelConfig(PretrainedConfig):
    model_type = "mymodel"  # 唯一标识名
    def __init__(self, vocab_size=50000, max_context=1024, embedding_dim=768, layers=12, head_num=12, dropout=0.0, pad_token_id=2, **kwargs):
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
        self.output = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

    def forward(self, input, target=None, **kwargs):
        embedding = self.token_embedding(input)
        embedding = self.drop_emb(embedding)
        for layer in self.transformer_layers:
            embedding = layer(embedding)
        embedding = self.norm(embedding)
        embedding = self.output(embedding)
        
        loss = None
        if target is not None:
            loss = torch.nn.functional.cross_entropy(embedding.flatten(0, 1), target.flatten(), ignore_index=self.config.pad_token_id)
        return {"loss": loss, "logits": embedding}

def generate_text(model, idx, max_tokens:int, max_context:int):
    for _ in range(max_tokens):
        idx_cond = idx[-max_context:]
        with torch.no_grad():
            logits = model(idx_cond)['logits']

        logits = logits[-1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=0)
    return idx