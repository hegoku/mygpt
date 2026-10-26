记录在 `RTX 4070 SUPER (12GB)` 上训练 `layers=24, embedding_dim=512, head_num=8, max_context=1024, 词表大小是 65,005` 的 `MyLlamaPretrainedModel` 模型的内存和性能的改进点

训练数据集是: [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)

在不进行下列优化的时候，当训练的配置是:

```python
num_train_epochs=1,
per_device_train_batch_size=7,
gradient_accumulation_steps=32,
eval_strategy="no",
learning_rate=0.0003,
logging_steps=5,
save_steps=500,
save_strategy="steps",
save_total_limit=3,
report_to="wandb",
weight_decay=0.1,
adam_beta2=0.95,
warmup_steps=0.03,
lr_scheduler_type="cosine",
optim="adamw_torch",
dataloader_pin_memory=False,
torch_compile=True,
remove_unused_columns=False
```

并且 new model 的时候 `to(device, dtype=bf16)`

训练性能:
- 显存: `12.5GB`
- 速度: `3.16s/it`
- 训练时间: `6小时51分`

按照下列顺序进行优化后的内存占用大小:

1. [训练时不返回 `logits`](#一训练时不返回-logits-以降低大词表下的峰值显存):
  - 显存: `10.1GB`
  - 速度: `6.82s/it`
  - 训练时间: `4小时9分`
2. [使用 `scaled_dot_product_attention`](#二使用-scaled_dot_product_attention-优化显存与训练速度):
  - 显存: `7.6GB`
  - 速度: `4.72s/it`
  - 训练时间: `2小时59分`
3. [使用 `LigerFusedLinearCrossEntropyLoss`](#三使用-ligerfusedlinearcrossentropyloss-消除大词表输出层显存峰值):
  - 显存: `5.8GB`
  - 速度: `7.25s/it`
  - 训练时间: `3小时35分`
4. bf16=True, tf32=True, 不`to(device, dtype=bf16)`:
  - 只开启 1和2，没开3的情况下
  - 显存: `9.2GB`
  - 速度: `6.19s/it`
  - 训练时间: `3小时51分`
  - loss比在只开启`1`和`2`的情况下小 `0.18`
5. bf16=True, tf32=True, 不`to(device, dtype=bf16)`:
  - 开3的情况下
  - 显存: `7.1GB`
  - 速度: `9.65s/it`
  - 训练时间: `6小时13分`
  - loss比在开启`1`,`2`,`3`的情况下小 `0.19`

---

# 一、训练时不返回 `logits` 以降低大词表下的峰值显存

## 解决的问题

在使用较大词表（例如 `vocab_size = 65005`）和长序列（`seq_len = 1024`）训练时，模型最后一层输出的 `logits` 张量极其庞大。如果 `forward` 在计算完 `loss` 后仍然把完整的 `logits` 放进 `CausalLMOutputWithPast` 返回给 HuggingFace `Trainer`：
1. **导致反向传播期间峰值显存暴涨（占用约 0.9GB ~ 1.8GB 显存）**：在 12GB 显存的 RTX 4070 SUPER 上极易触发显存溢出（OOM）或被迫借用系统内存。
2. **返回无用数据**：训练阶段（以及仅计算验证集 Loss 时）外部只需要标量 `loss`，根本不需要 `(batch_size, seq_len, vocab_size)` 的 `logits`；只有在推理生成（`labels is None`）时才需要 `logits`。

通过将训练时的返回值改为 `logits=None`，可以在**完全不影响训练和推理行为**的前提下，在反向传播开始瞬间立即释放掉近 1GB ~ 2GB 的输出层显存。

## 代码改动 (`MyLlamaPretrainedModel.py`)

在 `MyLlamaPretrainedModel.forward` 的返回语句中，根据是否传入 `labels` 动态决定是否返回 `logits`：

```python
        # ❌ 原有写法（无论训练还是推理，始终返回庞大的 logits 张量）：
        # return CausalLMOutputWithPast(loss=loss, logits=embedding, past_key_values=None)

        # ✅ 修改后（传入 labels 计算 loss 时不保留 logits 引用，仅在推理时返回 logits）：
        return CausalLMOutputWithPast(
            loss=loss,
            logits=None if labels is not None else embedding,
            past_key_values=None,
        )
```

## 原理解析

### 1. 大词表下的 `logits` 张量到底有多大？
以当前训练配置（`batch_size = 7`、`seq_len = 1024`、`vocab_size = 65005`）为例，单次前向传播最后一层线性投影产生的 `logits`（代码中的 `embedding`）元素数量高达：
$$7 \times 1024 \times 65005 = \mathbf{465,955,840 \text{（约 4.66 亿个浮点数）}}$$
* 在 **`bfloat16`（每个元素 2 Bytes）** 下，仅这一个张量就占用 **$932\text{ MB} \approx 0.91\text{ GB}$** 显存。
* 如果开启混合精度 **`bf16=True`**，`cross_entropy` 还会产生对应的 **`float32`（每个元素 4 Bytes）** 拷贝及梯度，峰值显存占用高达 **$1.86\text{ GB} \sim 3.72\text{ GB}$**。

### 2. Python 引用计数与 PyTorch 反向传播显存释放机制
在 HuggingFace `Trainer.training_step` 中，执行顺序是：
```python
outputs = model(**inputs)      # 1. 前向传播，拿到 CausalLMOutputWithPast
loss = outputs.loss            # 2. 取出 loss
self.accelerator.backward(loss) # 3. 执行整网反向传播（从第 24 层一路回传到第 1 层）
```
* **修改前（`logits=embedding`）**：
  因为 `outputs.logits` 一直持有对这 `4.66` 亿个元素的张量引用，只要 `training_step` 函数还没执行完（即在整个 24 层 Transformer 漫长的 `backward` 过程中），Python 的引用计数就不为 0，**PyTorch 显存分配器绝对不敢回收这 0.91GB ~ 1.86GB 的显存**。它被迫与反向传播过程中新产生的各层梯度叠加在一起，直接推高了峰值显存（Peak VRAM）。
* **修改后（`logits=None if labels is not None else embedding`）**：
  当 `forward` 函数执行完毕退出作用域时，局部变量 `embedding` 被销毁，外部的 `outputs.logits` 为 `None`。此时只有计算图（Autograd Graph）的 `CrossEntropyBackward` 节点内部临时持有它；一旦 `backward(loss)` 开始执行，**第一步算完输出层 Cross-Entropy 的梯度后，该张量就会被 PyTorch 立即释放并交还给显存池**，供后续 24 层 Transformer 的反向传播复用！

---

# 二、使用 `scaled_dot_product_attention` 优化显存与训练速度

## 解决的问题

在 RTX 4070 SUPER (12GB) 上训练 24 层、`max_context=1024` 的模型时，如果开启 `bf16=True`，原有的手写 Attention 会导致显存超过 12GB（溢出到系统内存，导致训练速度大幅变慢）。

改用 `torch.nn.functional.scaled_dot_product_attention` (SDPA) 可以直接解决以下问题：
1. **大幅减少显存占用（节省约 3.5~4GB 显存）**：避免在 12GB 显存下开启 `bf16=True` 时发生显存溢出（OOM）或借用系统内存。
2. **大幅提升训练速度（提速 2~3 倍）**：减少 GPU 显存（VRAM）的反复读写，并自动跳过因果掩码（上三角区域）的一半无效计算。
3. **无损替换，不影响模型训练效果**：计算结果与手写代码在数学上 **100% 严格等价**，不会损失任何精度或模型性能。

## 代码改动 (`Attention.py`)

在 `Attention.py` 的 `LlamaMultiHeadAttention.forward`（以及 `MultiHeadAttention.forward`）中，将手写的注意力矩阵计算替换为 `nn.functional.scaled_dot_product_attention`：

```python
        q = compute_rope(q, self.cos, self.sin)
        k = compute_rope(k, self.cos, self.sin)

        # ❌ 原有的手写实现（每层都会在显存中生成并保存 1024x1024 的大矩阵）：
        # mask = self.mask[:token_num, :token_num]
        # if padding_mask is not None:
        #     mask = mask | padding_mask
        #
        # scores = q @ k.transpose(-2, -1)
        # scores.masked_fill_(mask, -torch.inf)
        # weights = torch.softmax(scores / self.softmax_factor, dim=-1)
        #
        # # Shape: (b, num_tokens, num_heads, head_dim)
        # output = (weights @ v).transpose(-3, -2)

        # ✅ 改用 PyTorch 原生 SDPA (自动调用 FlashAttention-2)：
        output = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
        )
        output = output.transpose(-3, -2)

        # Combine heads, where self.embedding_size = self.head_num * self.d_q
        output = output.reshape(*output.shape[:-2], self.embedding_size)
        output = self.output(output)
        return output
```

> **附注**：启用 `is_causal=True` 后，`init_buffers()` 中预先生成的上三角 `mask` 也不再需要，可以直接省去 `self.register_buffer("mask", mask, persistent=False)`。

## 原理解析

### 1. 为什么手写 Attention 极其消耗显存？
手写代码分四步执行：
1. `scores = q @ k.transpose(-2, -1)`：生成形状为 `(batch, 8, 1024, 1024)` 的矩阵并写回显存。
2. `scores.masked_fill_(mask, -torch.inf)`：掩码填充。
3. `weights = torch.softmax(scores / self.softmax_factor, dim=-1)`：再次生成 `(batch, 8, 1024, 1024)` 的矩阵并写回显存。
4. `output = weights @ v`：乘以 `v` 得到最终输出 `(batch, 8, 1024, 64)`。

虽然我们最终只需要 `1024 × 64` 的 `output`，但为了后续反向传播（`backward`）求梯度，PyTorch **必须将全部 24 层的 `1024 × 1024` 中间注意力矩阵一直保存在显存中**，显存复杂度为 $O(N^2)$，光这一项就占用了 3.5~4GB 显存。

### 2. 为什么 `scaled_dot_product_attention` 能大幅省显存？
`scaled_dot_product_attention` 在 RTX 4070 SUPER 上会自动调用底层的 **FlashAttention-2** 内核，核心机制是 **分块计算（Tiling）+ 在线 Softmax（Online Softmax）**：
* **不写回中间大矩阵**：每次只从慢速显存（VRAM）读取一小块 `q, k, v` 到 GPU 超高速的片上缓存（SRAM）中，在缓存内部一口气算完 `Q @ K.T -> 除以缩放因子 -> 因果掩码 -> Softmax -> 乘以 V`，直接把中间注意力矩阵丢弃，只把最终很小的一块 `output`（`1024 × 64`）写回显存。
* **反向传播重计算**：反向传播时不在显存中存储那 24 个 `1024 × 1024` 矩阵，而是利用高速片上缓存现场重算，使注意力层的显存占用从 $O(N^2)$ 降为 $O(N)$（中间矩阵显存缩小 16 倍）。

### 3. 计算结果和手写版一致吗？会影响模型性能吗？
* **数学上 100% 精确等价**：FlashAttention 是**精确注意力算法（Exact Attention）**，不是近似或稀疏算法。它利用指数运算法则 $e^{x - m_{\text{new}}} = e^{x - m_{\text{old}}} \cdot e^{m_{\text{old}} - m_{\text{new}}}$ 将全局 Softmax 拆解为分块递推更新，最终算出的结果与手写版完全一致：
  * 手写的 `/ self.softmax_factor` $\Longleftrightarrow$ SDPA 内部自动除以 $\sqrt{d_q}$。
  * 手写的 `scores.masked_fill_(mask, -torch.inf)` $\Longleftrightarrow$ SDPA 的 `is_causal=True`。
  * 手写的 `torch.softmax(...) @ v` $\Longleftrightarrow$ SDPA 内部融合计算。
* **数值精度反而更高**：手写版在 `bfloat16` 下每一步中间结果写回显存都会经历一次 `bfloat16` 截断（共 3 次舍入误差）；而 SDPA 在 GPU 寄存器内部全程使用 **`float32` 累加器**完成全部计算，仅在最终输出时转回 `bfloat16`（仅 1 次舍入误差），因此数值稳定性和训练效果更好。

---

# 三、使用 `LigerFusedLinearCrossEntropyLoss` 消除大词表输出层显存峰值

## 解决的问题

即使在第一项优化中将返回值改成了 `logits=None`（让反向传播算完输出层后能立即释放 `logits`），在**前向传播计算 `F.linear` 到反向传播第一步计算 `CrossEntropyBackward` 之间**，完整的 `(batch_size, seq_len, vocab_size)` 张量依然必须在显存中短暂实例化：
1. **输出层瞬时峰值显存极高（尤其在开启 `bf16=True` 混合精度时）**：当 `vocab_size = 65005` 时，`F.linear` 产出的 `logits` 有 4.66 亿个元素。在 `bf16=True`（AMP）下，PyTorch 为了数值稳定会自动把 `logits` 升级（Upcast）为 `float32` 再计算 `cross_entropy`，瞬间产生 `bfloat16 logits (0.91GB) + float32 logits (1.86GB) + float32 梯度 (1.86GB)`，峰值显存高达 **3GB ~ 4GB**，直接导致 12GB 显存溢出。
2. **与 `torch.compile` 结合时的 Graph Break 问题**：`LigerFusedLinearCrossEntropyLoss` 内部使用 Triton 自定义算子并调用了 `Tensor.item()`，如果直接在 `forward` 中调用会触发 `torch._dynamo` 的 `Graph break from Tensor.item()` 警告。

使用 `LigerFusedLinearCrossEntropyLoss` 并配合 `@torch._dynamo.disable`，可以**彻底避免在显存中生成完整的 `logits` 张量**，将输出层的显存占用从数 GB 降低到几十 MB，同时完美兼容 `torch_compile=True`。

## 代码改动 (`MyLlamaPretrainedModel.py`)

为了同时兼容支持 `liger-kernel` 的 CUDA 机器和不支持它的环境（如 macOS MPS / CPU 或未安装 `liger-kernel` 的机器），并支持在显存充足时手动关闭，采用以下跨平台兼容写法：

1. 使用 `try...except ImportError` 安全导入 `LigerFusedLinearCrossEntropyLoss`，并在 `MyLlamaPretrainedModelConfig` 中增加 `use_liger_loss: bool = True` 开关：

```python
try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    HAS_LIGER_KERNEL = True
except ImportError:
    LigerFusedLinearCrossEntropyLoss = None
    HAS_LIGER_KERNEL = False

class MyLlamaPretrainedModelConfig(PretrainedConfig):
    model_type = "myllama2"
    def __init__(self, vocab_size=50000, max_context=1024, embedding_dim=768, layers=12, head_num=12, ff_dim=2048, pad_token_id=-100, rope_theta:float=10000.0, use_liger_loss:bool=True, **kwargs):
        super().__init__(**kwargs)
        ...
        self.use_liger_loss = use_liger_loss
```

2. 在 `__init__` 中按需初始化 Liger Loss 实例，并在 `forward` 中同时判断 **`self.loss_fn_sum is not None`、`embedding.is_cuda`（必须是 CUDA 设备，macOS 的 `mps` 会自动走 `else` 分支）以及 `config.use_liger_loss`**：

```python
class MyLlamaPretrainedModel(PreTrainedModel, GenerationMixin):
    def __init__(self, config):
        ...
        self.norm = nn.RMSNorm(config.embedding_dim, eps=1e-6)
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

        loss = None
        logits = None
        if labels is not None:
            num_items_in_batch = kwargs.get("num_items_in_batch", None)
            use_liger = (
                self.loss_fn_sum is not None
                and embedding.is_cuda
                and getattr(self.config, "use_liger_loss", True)
            )
            if use_liger:
                # ✅ CUDA + 已安装 liger_kernel：走 Triton 分块融合算子，不生成完整 logits
                loss = self._compute_liger_loss(
                    embedding.reshape(-1, self.embedding_dim),
                    labels.reshape(-1),
                    num_items_in_batch,
                )
            else:
                # ✅ macOS (MPS) / CPU / 手动关闭 use_liger_loss：回退到原生 PyTorch 实现（同样不返回 logits）
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
                        reduction="mean",
                    )
        else:
            # ✅ 仅在推理生成（labels 为 None）时才计算并返回完整的 logits
            logits = F.linear(embedding, self.token_embedding.weight)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
        )
```

## 原理解析

### 1. 为什么传统 `F.linear + F.cross_entropy` 是显存瓶颈？
在 Transformer 最后一层，`norm` 之后的隐状态 `embedding` 形状仅为 `(7, 1024, 512)`（在 `bfloat16` 下仅占 **7 MB** 显存）。
但一旦执行 `F.linear(embedding, self.token_embedding.weight)` 投影到 `65005` 的词表维度，张量形状立刻变成 `(7, 1024, 65005)`（体积暴涨 **127 倍**）。为了求导，PyTorch 必须保留完整的 `logits` 张量直到 `CrossEntropyBackward` 算完。

### 2. `LigerFusedLinearCrossEntropyLoss` 的分块融合机制（Chunked Fused Kernel）
Liger-Kernel 将 **线性投影（`Linear`）+ `Softmax` + 交叉熵损失（`CrossEntropy`）+ 反向梯度计算** 融合成了一个基于 OpenAI Triton 的算子：
1. **对 Token 维度分块（Chunking）**：算子不会一次性计算全部 $B \times L = 7168$ 个 token 的 `logits`，而是将其切成很小的块（Chunk），每次只拿一小块隐状态与 `token_embedding.weight` 相乘。
2. **前向与梯度在片上原地完成（In-place Gradient Computation）**：由于交叉熵对 `logits` 的导数公式极其简洁（$\frac{\partial \mathcal{L}}{\partial z_i} = \text{softmax}(z)_i - \mathbb{I}(i = y)$），Liger 算子在算出每一小块 `logits` 和 `loss` 的**同一时刻**，直接在 GPU 高速缓存中算出该块的梯度，并累加回传给 `hidden_states` 和 `token_embedding.weight`，随后**立即丢弃这一小块 `logits`**。
3. **0 精度损失**：整个分块计算与在线 Softmax 递推在数学上与原生 PyTorch **100% 严格等价**，但全程没有任何完整的 `(B, L, V)` 张量驻留在显存中。

### 3. 为什么要加 `@torch._dynamo.disable`？
* 当 `TrainingArguments` 开启 `torch_compile=True` 时，PyTorch Dynamo 会尝试把整个 `forward` 追踪编译成一张静态计算图。
* 而 `LigerFusedLinearCrossEntropyLoss` 本身已经是手写调优好的底层 Triton `autograd.Function`，其内部调用了 `Tensor.item()` 来读取有效 token 数，这会打断 Dynamo 的图捕获并抛出 `Graph break` 警告。
* 给 `_compute_liger_loss` 加上 `@torch._dynamo.disable` 后，前面的 `Embedding + 24 层 Transformer + RMSNorm` 会被 `torch.compile` 完整编译为一个无中断的高效计算图，而最后一步的 Loss 计算则直接无缝移交给 Liger 的 Triton 内核原生执行。

### 4. 实测结论：为什么在小 Batch（显存未满）下开启 `LigerFusedLinearCrossEntropyLoss` 训练速度反而会变慢？
在已启用优化 1（不返回 `logits`）和优化 2（SDPA）后，当 `per_device_train_batch_size=7` 时显存占用仅为 `7.6GB`（远低于 12GB 上限）。此时若直接开启 `LigerFusedLinearCrossEntropyLoss`，虽然显存进一步下降，但**单步训练速度反而变慢**，原因在于典型的**“空间换时间”**开销：

1. **分块重计算（Recomputation）不及单次 `cuBLAS` 大矩阵乘快**：
   * 原生 `F.linear` 调用的是 NVIDIA 闭源极致调优的 `cuBLAS` 算子，一次性完成 `7168 × 512 @ 512 × 65005` 的大矩阵乘法，GPU 满载率极高。
   * 而 Liger 为了不保存 `logits`，必须在 `forward` 阶段将输入切成多个小块（Chunks），在循环中执行多次小矩阵乘法，并立即在前向阶段额外计算 $dX$ 和 $dW$ 梯度及片上原子累加。在显存本就充裕时，多次分块小矩阵乘 + 重计算比单次 `cuBLAS` 慢。
2. **内部 `Tensor.item()` 触发了 CPU-GPU 强制同步阻塞（CUDA Synchronization）**：
   * `LigerFusedLinearCrossEntropyLoss` 内部为了统计非 padding 的有效 token 数，执行了 `(target != ignore_index).sum().item()`。
   * `.item()` 会强制把标量从 GPU 显存拷贝回 CPU 内存，触发 `cudaStreamSynchronize`，迫使 CPU 原地死等 GPU 把前面 24 层 Transformer 全部算完，打断了 CPU 异步下发 Kernel 的流水线。在 `gradient_accumulation_steps=32` 下，每个优化步要被迫停顿同步 32 次。
3. **打断了 `torch.compile` 的尾部算子融合**：
   * 原生 `F.linear + F.cross_entropy` 可以和前面的 `RMSNorm` 一起留在 `torch.compile` (`inductor`) 的同一张静态计算图里一气呵成；而切换到外部 Liger 算子会在每个 micro-batch 尾部产生一次退出编译图再回到 Python 调度的开销。
4. **何时应该开启它？（适用场景总结）**：
   * **显存够用时（如纯 `bf16` 权重 + `batch_size=7/8`，显存仅 7.6GB）**：在初始化 `MyLlamaPretrainedModelConfig(..., use_liger_loss=False)` 关闭它，直接用原生 `F.linear + F.cross_entropy` 速度最快。
   * **显存不够用时（如开启 `TrainingArguments(bf16=True)` 混合精度，或想把 `per_device_train_batch_size` 从 `7` 翻倍拉大到 `16` 时）**：原生实现会因 `logits` 占满 12GB 显存而溢出到系统内存导致速度暴跌，此时开启 `use_liger_loss=True` 省下 2~3GB 显存即可让大 Batch 在显存内满速运行。


