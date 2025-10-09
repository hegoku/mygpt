import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import os
import Tokenizer
import MyDataset
import MLXLlama
import MLXTrainer
import DeepseekTokenizer
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from datasets import load_from_disk
from torchdata.stateful_dataloader import StatefulDataLoader

mx.random.seed(123)

file_path = 'train_data.txt'

text_data = ''
with open(file_path, "r", encoding="utf-8") as file:
	text_data = file.read()

train_ratio = 0.80
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

# tokenizer = Tokenizer.Tokenizer()
tokenizer = DeepseekTokenizer.DeepseekTokenizer()
model = MLXLlama.MLXLlama(tokenizer=tokenizer, layer=12 , max_context=1024, embedding_dim=768, head_num=12)
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params:,}")

# data_path = 'fjcanyue/wikipedia-zh-cn'
# dataset = load_dataset(data_path, split='train[0:1000]')
# a = MyDataset.MyDataset2(dataset, tokenizer, 1024, 1024)

# train_loader = DataLoader(
#    a,
#    batch_size=8,
#    shuffle=True,
#    drop_last=True,
#    num_workers=0
# )
# print("train data loaded")

# dataset = load_dataset(data_path, split='train[1000:1150]')
# b = MyDataset.MyDataset2(dataset, tokenizer, 1024, 1024)
# val_loader = DataLoader(
#    b,
#    batch_size=4,
#    shuffle=False,
#    drop_last=False,
#    num_workers=0
# )
# print("val data loaded")

# train_loader = MyDataset.create_dataloader_v1(train_data, tokenizer, max_length=50, stride=20, drop_last=True, shuffle=True)
# val_loader = MyDataset.create_dataloader_v1(val_data, tokenizer, max_length=50, stride=20, drop_last=False, shuffle=False)
# print("Train loader:")
# for x, y in train_loader:
#     print(x.shape, y.shape)

# print("\nValidation loader:")
# for x, y in val_loader:
#     print(x.shape, y.shape)

# aa = mx.array([[1,2,3],[2,4,2], [5,2,6]])
# active_tokens = (aa.flatten() != tokenizer.pad_token_id)
# print((aa.flatten() * active_tokens).sum())
# print(active_tokens)

train_dataset = load_from_disk('./train_wiki_1024')
train_dataset = train_dataset.with_format("numpy")
train_loader = StatefulDataLoader(train_dataset, batch_size=5, shuffle=True, drop_last=False)
print("train data loaded")

val_dataset = load_from_disk('./val_wiki_1024')
val_dataset = val_dataset.with_format("numpy")
val_loader = StatefulDataLoader(val_dataset, batch_size=1, shuffle=True)
print("val data loaded")

model.set_dtype(mx.bfloat16)

optimizer = optim.AdamW(learning_rate=0.0004, weight_decay=0.1, betas=[0.9, 0.98])
num_epochs = 10
train_losses, val_losses, tokens_seen = MLXTrainer.train_model_simple(
    model, train_loader, val_loader, optimizer, 
    num_epochs=num_epochs, eval_freq=5, eval_iter=4,
    start_context="例如在西非", tokenizer=tokenizer, save_dir="./checkpoint", save_freq=1000
)