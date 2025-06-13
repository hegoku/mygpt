import torch
import os
import Tokenizer
import MyDataset
import MyGPT
import Trainer
import DeepseekTokenizer

torch.manual_seed(123)

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
model = MyGPT.MyGPT(tokenizer=tokenizer, layer=12 , max_context=1024, embedding_dim=768, d_q=64, d_v=64, dropout=0.1, head_num=12)

train_loader = MyDataset.create_dataloader_v1(train_data, tokenizer, max_length=50, stride=10, drop_last=True, shuffle=True)
val_loader = MyDataset.create_dataloader_v1(val_data, tokenizer, max_length=50, stride=10, drop_last=False, shuffle=False)
# print("Train loader:")
# for x, y in train_loader:
#     print(x.shape, y.shape)

# print("\nValidation loader:")
# for x, y in val_loader:
#     print(x.shape, y.shape)
    
if torch.cuda.is_available():
   device = torch.device("cuda")
elif torch.backends.mps.is_available():
   device = torch.device("mps")
else:
   device = torch.device("cpu")

print(f"Using {device} device.")


model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 100
train_losses, val_losses, tokens_seen = Trainer.train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="QKV计算", tokenizer=tokenizer
)