from transformers import Trainer, TrainingArguments, trainer_pt_utils
# import bitsandbytes as bnb
import torch
import MyLlama

ALL_LAYERNORM_LAYERS = [torch.nn.LayerNorm, torch.nn.RMSNorm]

class MyHuggingfaceTrainer(Trainer):
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
            pad_token_id = getattr(self.model.config, "pad_token_id", -100)
            num_items_in_batch = sum(
                ((batch["labels"] != -100) & (batch["labels"] != pad_token_id)).sum()
                for batch in batch_samples
            )
            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(device)
            return num_items_in_batch
        return super()._get_num_items_in_batch(batch_samples, device)
