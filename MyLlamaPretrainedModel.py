import os
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")
os.environ.setdefault("OTEL_TRACES_EXPORTER", "none")
os.environ.setdefault("LIGER_DISABLE_TELEMETRY", "1")

from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
import Attention
import MyLlama
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    HAS_LIGER_KERNEL = True
except ImportError:
    LigerFusedLinearCrossEntropyLoss = None
    HAS_LIGER_KERNEL = False

class MyLlamaPretrainedModelConfig(PretrainedConfig):
    model_type = "myllama2"  # 唯一标识名
    def __init__(self, vocab_size=50000, max_context=1024, embedding_dim=768, layers=12, head_num=12, ff_dim=2048, pad_token_id=-100, rope_theta:float=10000.0, use_liger_loss:bool=True, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.pad_token_id = pad_token_id
        self.layers = layers
        self.head_num = head_num
        self.ff_dim = ff_dim
        self.max_context = max_context
        self.rope_theta = rope_theta
        self.use_liger_loss = use_liger_loss

class MyLlamaPretrainedModel(PreTrainedModel, GenerationMixin):
    config_class = MyLlamaPretrainedModelConfig  # 关联配置

    def __init__(self, config: MyLlamaPretrainedModelConfig):
        super().__init__(config)
        self.embedding_dim = config.embedding_dim
        self.max_context = config.max_context
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_token_id)
        self.transformer_layers = nn.ModuleList([MyLlama.TransformerBlock(config.max_context, config.embedding_dim, config.head_num, config.ff_dim, config.rope_theta) for _ in range(config.layers)])
        self.norm = nn.RMSNorm(config.embedding_dim, eps=1e-6)
        # self.output = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        if HAS_LIGER_KERNEL and getattr(config, "use_liger_loss", True):
            self.loss_fn_sum = LigerFusedLinearCrossEntropyLoss(
                ignore_index=config.pad_token_id,
                reduction="sum",
            )
            self.loss_fn_mean = LigerFusedLinearCrossEntropyLoss(
                ignore_index=config.pad_token_id,
                reduction="mean",
            )
        else:
            self.loss_fn_sum = None
            self.loss_fn_mean = None

        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, (Attention.LlamaMultiHeadAttention, Attention.MultiHeadAttention)):
            module.init_buffers()

    def get_input_embeddings(self):
        return self.token_embedding

    def set_input_embeddings(self, new_embeddings):
        self.token_embedding = new_embeddings

    # 使用 LigerFusedLinearCrossEntropyLoss 将线性投影 (F.linear) 与交叉熵 (cross_entropy) 融合分块计算，
    # 避免在显存中生成完整的 (batch_size, seq_len, vocab_size) 庞大 logits 张量，大幅降低峰值显存。
    # 加上 @torch._dynamo.disable 是因为 Liger 底层已经是手写好的 Triton 算子且内部调用了 Tensor.item()，
    # 跳过 Dynamo 编译可避免触发 Graph break 警告，同时不影响前面 Transformer 层的完整图编译。
    @torch._dynamo.disable
    def _compute_liger_loss(self, hidden_states, labels, num_items_in_batch=None):
        if num_items_in_batch is not None:
            return self.loss_fn_sum(
                self.token_embedding.weight,
                hidden_states,
                labels,
            ) / num_items_in_batch
        return self.loss_fn_mean(
            self.token_embedding.weight,
            hidden_states,
            labels,
        )

    def forward(self, input_ids, past_key_values=None, use_cache=None, labels=None, **kwargs):
        embedding = self.token_embedding(input_ids)
        for layer in self.transformer_layers:
            embedding = layer(embedding)
        embedding = self.norm(embedding)
        # embedding = self.output(embedding)

        loss = None
        # 训练/计算Loss时(labels不为None)不返回庞大的logits张量(B, L, V)，
        # 使PyTorch在算完cross_entropy梯度后能立即释放该显存，大幅降低反向传播的峰值显存占用；
        # 仅在推理生成时(labels为None)返回logits。
        logits = None
        if labels is not None:
            num_items_in_batch = kwargs.get("num_items_in_batch", None)
            use_liger = (
                self.loss_fn_sum is not None
                and embedding.is_cuda
                and getattr(self.config, "use_liger_loss", True)
            )
            if use_liger:
                loss = self._compute_liger_loss(
                    embedding.reshape(-1, self.embedding_dim),
                    labels.reshape(-1),
                    num_items_in_batch,
                )
            else:
                logits_for_loss = F.linear(embedding, self.token_embedding.weight)
                if num_items_in_batch is not None:
                    loss = F.cross_entropy(
                        logits_for_loss.flatten(0, 1),
                        labels.flatten(),
                        ignore_index=self.config.pad_token_id,
                        reduction="sum",
                    ) / num_items_in_batch
                else:
                    loss = F.cross_entropy(
                        logits_for_loss.flatten(0, 1),
                        labels.flatten(),
                        ignore_index=self.config.pad_token_id,
                    )
        else:
            logits = F.linear(embedding, self.token_embedding.weight)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": kwargs.get("attention_mask")
        }

def generate_text(model, idx, max_tokens:int, max_context:int, temperature=0.0, top_k=None, top_p=None, repetition_penalty=1.0, eos_id=None, tokenizer=None):
    for _ in range(max_tokens):
        idx_cond = idx[-max_context:]
        with torch.no_grad():
            logits = model(idx_cond)['logits']

        logits = logits[-1, :]

        # ✅ 重复性惩罚
        if repetition_penalty != 1.0:
            seen_tokens = set(idx.tolist())
            for token_id in seen_tokens:
                logits[token_id] /= repetition_penalty

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[-1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # Apply top_p (nucleus) filtering
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)

            # Create mask for tokens to remove
            sorted_mask = cumulative_probs > top_p
            # Ensure at least one token is kept
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False

            # Set logits of removed tokens to -inf
            logits[sorted_indices[sorted_mask]] = float('-inf')

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

        if tokenizer is not None:
            print(tokenizer.decode(idx_next, errors="ignore"), end='', flush=True)

        idx = torch.cat((idx, idx_next), dim=0)
    return idx