from transformers import PreTrainedModel, PretrainedConfig
import Attention
import MyGPT
import torch.nn as nn
from transformers import TrainingArguments
from transformers import Trainer
import DeepseekTokenizer
import MyDataset
import torch
from transformers import TrainerCallback

class MyModelConfig(PretrainedConfig):
    model_type = "mymodel"  # 唯一标识名
    def __init__(self, vocab_size=50000, max_context=1024, embedding_dim=768, layers=12, head_num=12, dropout=0.0, pad_token_id=2, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.pad_token_id = pad_token_id
        self.layers = layers
        self.head_num = head_num
        self.dropout = dropout
        self.max_context = max_context

class MyModel(PreTrainedModel):
    config_class = MyModelConfig  # 关联配置

    def __init__(self, config):
        super().__init__(config)
        self.embedding_dim = config.embedding_dim
        self.max_context = config.max_context
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_token_id)
        self.drop_emb = nn.Dropout(config.dropout)
        self.transformer_layers = nn.ModuleList([MyGPT.TransformerBlock(config.max_context, config.embedding_dim, config.head_num) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.embedding_dim, eps=1e-5)
        self.output = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

    def forward(self, input_ids, labels=None, **kwargs):
        embedding = self.token_embedding(input_ids)
        embedding = self.drop_emb(embedding)
        for layer in self.transformer_layers:
            embedding = layer(embedding)
        embedding = self.norm(embedding)
        embedding = self.output(embedding)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(embedding.view(-1, self.config.vocab_size), 
                            labels.view(-1))
        return {"loss": loss, "logits": embedding}


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=100,
    per_device_train_batch_size=4,
    # per_device_eval_batch_size=2,
    eval_strategy="no",
    learning_rate=0.0004,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=10,
    report_to="none",
    # weight_decay=0.1,
    # adam_beta2=0.98,
    seed=123,
    # lr_scheduler_type="cos"
)

if torch.cuda.is_available():
   device = torch.device("cuda")
elif torch.backends.mps.is_available():
   device = torch.device("mps")
else:
   device = torch.device("cpu")

print(f"Using {device} device.")

# 初始化模型
tokenizer = DeepseekTokenizer.DeepseekTokenizer()
config = MyModelConfig(vocab_size=tokenizer.len())
model = MyModel(config).to(device, dtype=torch.bfloat16)

torch.manual_seed(123)

file_path = 'train_data.txt'

text_data = ''
with open(file_path, "r", encoding="utf-8") as file:
	text_data = file.read()

train_ratio = 0.80
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

train_loader = MyDataset.MyDataset3(train_data, tokenizer, 1024, 1024, format="hg")

def generate_text(model, idx, max_tokens:int, max_context:int):
    for _ in range(max_tokens):
        idx_cond = idx[:, -max_context:]
        with torch.no_grad():
            res = model(idx_cond)

        logits = res['logits'][:,-1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    # context_size = model.pos_emb.weight.shape[0]
    # encoded = text_to_token_ids(start_context, tokenizer).to(device)
    encoded = tokenizer.encode(start_context).to(device)
    with torch.no_grad():
        token_ids = generate_text(
            model=model, idx=encoded.detach().clone().unsqueeze(0),
            max_tokens=30, max_context=model.max_context
        )
    # decoded_text = token_ids_to_text(token_ids, tokenizer)
    decoded_text = tokenizer.decode(token_ids[0])
    model.train()
    return decoded_text

class LossLoggingCallback(TrainerCallback):
    # def on_log(self, args, state, control, logs=None, **kwargs):
        # 每次记录日志时触发
        # if "loss" in logs and state.global_step % 50 == 0:
            # print(f"\nStep {state.global_step}: Loss = {logs['loss']:.4f}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        print(generate_and_print_sample(model, tokenizer, device, "数学家希望"))
        # 每个epoch结束时打印
        # print(f"\nEpoch {state.epoch} finished | "
            #   f"Average Loss: {state.log_history[-1]['loss']:.4f}")
        
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_loader,
    # eval_dataset=tokenized_dataset["validation"],
    callbacks=[LossLoggingCallback()]
)

trainer.train()  # 开始训练！

print(generate_and_print_sample(model, tokenizer, device, "数学家希望"))
