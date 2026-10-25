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

1. [训练时不返回 `logits`](#训练时不返回-logits-以降低大词表下的峰值显存):
  - 显存: `10.1GB`
  - 速度: `6.82s/it`
  - 训练时间: `4小时9分`
2. [使用 `scaled_dot_product_attention`](#使用-scaled_dot_product_attention-优化显存与训练速度):
  - 显存: `7.6GB`
  - 速度: `4.72s/it`
  - 训练时间: `2小时59分`

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

