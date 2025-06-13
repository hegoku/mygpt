import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import MyDataset
import MyGPT
import Trainer
import DeepseekTokenizer

torch.manual_seed(123)
if torch.cuda.is_available():
   device = torch.device("cuda")
elif torch.backends.mps.is_available():
   device = torch.device("mps")
else:
   device = torch.device("cpu")
print(f"Using {device} device.")

tokenizer = DeepseekTokenizer.DeepseekTokenizer()
model = MyGPT.MyGPT(tokenizer=tokenizer, layer=12 , max_context=1024, embedding_dim=768, d_q=768, d_v=768, dropout=0.1, head_num=12).to(device)

def encode_data(data):
    res = []
    for d in data['text']:
       res.append(tokenizer.encode(d))
    return {'text':res}

data_path = 'wikimedia/wikimedia'
dataset = load_dataset(data_path, split='train[0:10]')
a = MyDataset.MyDataset2(dataset, tokenizer, 100, 50)
train_loader = DataLoader(
   a,
   batch_size=4,
   shuffle=True,
   drop_last=True,
   num_workers=0
)
print("train data loaded")
# dataset = dataset.map(encode_data, batched=True)
# dataset.set_format(type='torch', columns=['text'])
# train_loader = torch.utils.data.DataLoader(dataset, batch_size=32)
# next(iter(train_loader))

dataset = load_dataset(data_path, split='train[10:20]')
b = MyDataset.MyDataset2(dataset, tokenizer, 100, 50)
val_loader = DataLoader(
   b,
   batch_size=4,
   shuffle=False,
   drop_last=False,
   num_workers=0
)
print("val data loaded")
# dataset = dataset.map(encode_data, batched=True)
# dataset.set_format(type='torch', columns=['text'])
# val_loader = torch.utils.data.DataLoader(dataset, batch_size=32)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 10
train_losses, val_losses, tokens_seen = Trainer.train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="例如在西非", tokenizer=tokenizer
)