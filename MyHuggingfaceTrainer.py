from transformers import Trainer, TrainingArguments, trainer_pt_utils
# import bitsandbytes as bnb
import torch
import MyLlama

ALL_LAYERNORM_LAYERS = [torch.nn.LayerNorm, torch.nn.RMSNorm]

class MyHuggingfaceTrainer(Trainer):
    def __init__(self, *args, is_packed_dataset: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        # 是否为无缝拼接打包的数据集（如 g_data_v2.py 生成的数据，不含任何 pad_token_id）。
        # 若为 True，计算 batch 有效 token 数时无需逐元素比较，直接用 numel() O(1) 获取形状乘积。
        self.is_packed_dataset = is_packed_dataset

    def get_decay_parameter_names(self, model) -> list[str]:
        """
        Get all parameter names that weight decay will be applied to.

        This function filters out parameters in two ways:
        1. By layer type (instances of layers specified in ALL_LAYERNORM_LAYERS)
        2. By parameter name patterns (containing 'bias', 'layernorm', or 'rmsnorm')
        """
        decay_parameters = trainer_pt_utils.get_parameter_names(model, ALL_LAYERNORM_LAYERS, ["bias", "layernorm", "rmsnorm"])
        return decay_parameters

    def _get_num_items_in_batch(self, batch_samples: list, device: torch.device):
        if len(batch_samples) > 0 and "labels" in batch_samples[0] and self.model_accepts_loss_kwargs:
            # 如果是无缝拼接打包的数据集（没有 padding），所有 labels 均为有效 token，
            # 直接通过 .numel() 获取每个 batch 的元素总数（即 batch_size * seq_len），
            # 避免在 CPU 上对几十万个元素做布尔矩阵比较和求和，消除每个 global_step 开头的 CPU 阻塞。
            if self.is_packed_dataset:
                total_tokens = sum(batch["labels"].numel() for batch in batch_samples)
                return torch.tensor(total_tokens, device=device)

            pad_token_id = getattr(self.model.config, "pad_token_id", -100)
            num_items_in_batch = sum(
                ((batch["labels"] != -100) & (batch["labels"] != pad_token_id)).sum()
                for batch in batch_samples
            )
            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(device)
            return num_items_in_batch
        return super()._get_num_items_in_batch(batch_samples, device)
