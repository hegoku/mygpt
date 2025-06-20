import torch
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader
import MyDataset
import MyGPT
import MyLlama
import Trainer2
import DeepseekTokenizer
import os
from torchdata.stateful_dataloader import StatefulDataLoader

torch.manual_seed(123)
if torch.cuda.is_available():
   device = torch.device("cuda")
elif torch.backends.mps.is_available():
   device = torch.device("mps")
else:
   device = torch.device("cpu")
print(f"Using {device} device.")

tokenizer = DeepseekTokenizer.DeepseekTokenizer()
# model = MyGPT.MyGPT(tokenizer=tokenizer, layer=12 , max_context=512, embedding_dim=768, d_q=64, d_v=64, dropout=0.1, head_num=12).to(device=device, dtype=torch.bfloat16)
model = MyLlama.MyLlama(tokenizer=tokenizer, layer=12 , max_context=1024, embedding_dim=768, head_num=12).to(device=device, dtype=torch.bfloat16)

train_dataset = load_from_disk('./train_wiki')
# train_dataset = train_dataset.to_iterable_dataset()
# train_dataset = train_dataset.shuffle(123, buffer_size=100)
# train_dataset = train_dataset.batch(batch_size=16)
train_dataset = train_dataset.with_format("torch")
train_loader = StatefulDataLoader(train_dataset, batch_size=5, shuffle=True)
print("train data loaded")

# val_dataset = load_from_disk('./val_wiki')
# val_loader = val_loader.to_iterable_dataset()
# val_dataset = val_dataset.batch(batch_size=16)
# val_dataset = val_dataset.with_format("torch")
val_loader = StatefulDataLoader(train_dataset, batch_size=5, shuffle=True)
print("val data loaded")

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 50
train_losses, val_losses, tokens_seen = Trainer2.train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=100, eval_iter=5,
    start_context="例如在西非", tokenizer=tokenizer, write_log=True, save_dir="./checkpoint", save_freq=30000
)

# epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
# Trainer.plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)