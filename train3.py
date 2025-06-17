import torch
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader
import MyDataset
import MyGPT
import Trainer2
import DeepseekTokenizer
import os

torch.manual_seed(123)
if torch.cuda.is_available():
   device = torch.device("cuda")
elif torch.backends.mps.is_available():
   device = torch.device("mps")
else:
   device = torch.device("cpu")
print(f"Using {device} device.")

tokenizer = DeepseekTokenizer.DeepseekTokenizer()
model = MyGPT.MyGPT(tokenizer=tokenizer, layer=12 , max_context=512, embedding_dim=768, d_q=64, d_v=64, dropout=0.1, head_num=12).to(device=device, dtype=torch.bfloat16)

def encode_data(data):
    res = []
    for d in data['text']:
       res.append(tokenizer.encode(d))
    return {'text':res}

train_loader = load_from_disk('./train_wiki')
train_loader = train_loader.to_iterable_dataset()
train_loader = train_loader.shuffle(123, buffer_size=100)
train_loader = train_loader.batch(batch_size=16)
train_loader = train_loader.with_format("torch")
# train_loader = MyDataset.MyDataset3(dataset, tokenizer, 512, 512)
print("train data loaded")

val_loader = load_from_disk('./val_wiki')
val_loader = val_loader.to_iterable_dataset()
val_loader = val_loader.batch(batch_size=16)
val_loader = val_loader.with_format("torch")
# val_loader = MyDataset.MyDataset3(dataset_v, tokenizer, 512, 512)
print("val data loaded")

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 50
train_losses, val_losses, tokens_seen = Trainer2.train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="例如在西非", tokenizer=tokenizer, write_log=True
)

# epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
# Trainer.plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)