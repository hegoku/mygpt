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
import os
from datasets import load_from_disk
import MyGPTPretrainedModel
import MyLlamaPretrainedModel
from torch.utils.data import Dataset, DataLoader
import MyHuggingfaceTrainer

os.environ["WANDB_PROJECT"] = 'mygpt'

training_args = TrainingArguments(
    output_dir="./results1",
    num_train_epochs=1,
    per_device_train_batch_size=7,
    # ignore_data_skip=True,
    # per_device_eval_batch_size=2,
    gradient_accumulation_steps=72,
    eval_strategy="no",
    learning_rate=0.001,
    logging_dir="./logs",
    logging_steps=5,
    save_steps=500,
    save_strategy="steps",
    save_total_limit=3,
    report_to="wandb",
    weight_decay=0.1,
    adam_beta2=0.95,
    # seed=123,
    # warmup_ratio=0.03,
    warmup_steps=0,
    lr_scheduler_type="warmup_stable_decay",
    lr_scheduler_kwargs={
        "num_decay_steps":16804,
        "num_stable_steps":24178,
        "min_lr_ratio":0.01
    },
    run_name="test",
    optim="adamw_bnb_8bit",
    dataloader_pin_memory=False,
    torch_compile=True,
    label_names=['target'],
    remove_unused_columns=False
    # resume_from_checkpoint=True
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
# config = MyGPTPretrainedModel.MyGPTPretrainedModelConfig(vocab_size=tokenizer.len(), pad_token_id=tokenizer.pad_token_id)
# model = MyGPTPretrainedModel.MyGPTPretrainedModel(config).to(device, dtype=torch.bfloat16)
config = MyLlamaPretrainedModel.MyLlamaPretrainedModelConfig(vocab_size=tokenizer.len(), pad_token_id=tokenizer.pad_token_id, layers=24, embedding_dim=512, head_num=8, max_context=2048)
model = MyLlamaPretrainedModel.MyLlamaPretrainedModel(config).to(device, dtype=torch.bfloat16)
# model = MyLlamaPretrainedModel.MyLlamaPretrainedModel.from_pretrained("./results1/checkpoint-47000").to(device, dtype=torch.bfloat16)

train_dataset = load_from_disk('./aaa')
# train_dataset = train_dataset.to_iterable_dataset()
# train_dataset = train_dataset.shuffle(123, buffer_size=100)
# train_dataset = train_dataset.batch(batch_size=16)
train_dataset = train_dataset.with_format("torch")
# train_loader = StatefulDataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=False)
print("train data loaded")

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

def data_collator(batch):
    ids = torch.tensor([example['input'].tolist() for example in batch])
    # print(ids.shape, ids, ids[:,:-1], ids[:,1:])
    return {"input_ids":ids[:,:-1], "labels":ids[:,1:]}

trainer = MyHuggingfaceTrainer.MyHuggingfaceTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=tokenized_dataset["validation"],
    callbacks=[LossLoggingCallback()],
    data_collator=data_collator
)

trainer.train(resume_from_checkpoint="./results1/checkpoint-72500")  # 开始训练！
# trainer.train()

print(generate_and_print_sample(model, tokenizer, device, "数学家希望"))
