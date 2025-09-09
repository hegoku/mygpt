from transformers import Trainer, TrainingArguments
from torch.optim import AdamW
import torch

class MyHuggingfaceTrainer(Trainer):
    def create_optimizer(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight", "norm.weight", "embedding"]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]

        optimizer_kwargs = {
            "betas": (self.args.adam_beta1, self.args.adam_beta2),
            "eps": self.args.adam_epsilon,
            "lr": self.args.learning_rate,
        }
        self.optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

        if self.sharded_dpp:
            self.optimizer = self.sharded_dpp.create_optimizer(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer