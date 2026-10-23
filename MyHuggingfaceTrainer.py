from transformers import Trainer, TrainingArguments, trainer_pt_utils
import bitsandbytes as bnb
import torch
import MyLlama

ALL_LAYERNORM_LAYERS = [torch.nn.LayerNorm, torch.nn.Embedding, MyLlama.RMSNorm, torch.nn.RMSNorm]

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