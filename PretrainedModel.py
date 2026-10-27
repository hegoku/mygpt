import os
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["OTEL_METRICS_EXPORTER"] = "none"
os.environ["OTEL_TRACES_EXPORTER"] = "none"
os.environ["LIGER_DISABLE_TELEMETRY"] = "1"
os.environ["WANDB_PROJECT"] = 'mygpt'

from transformers import PreTrainedModel, PretrainedConfig
import Attention
import MyGPT
import torch.nn as nn
from transformers import TrainingArguments
from transformers import Trainer
import DeepseekTokenizer
import torch
from transformers import TrainerCallback
from datasets import load_from_disk
import MyGPTPretrainedModel
import MyLlamaPretrainedModel
from torch.utils.data import Dataset, DataLoader
import MyHuggingfaceTrainer
import math
import shutil

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
    def __init__(self, stable_end_step: int, model=None, tokenizer=None, device=None):
        self.stable_end_step = stable_end_step
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def on_step_end(self, args, state, control, **kwargs):
        # 无论 save_steps 是多少，只要到达 stable 结束的那一步，强制触发一次保存
        if state.global_step == self.stable_end_step:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        # 当 stable 结束那一步的 checkpoint 保存完成后，将它备份到独立目录，
        # 防止后续 Decay 阶段的 save_total_limit 把它当成旧 checkpoint 删掉！
        if state.global_step == self.stable_end_step:
            src_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            dst_dir = os.path.join(args.output_dir, f"stable-checkpoint-{state.global_step}")
            if os.path.exists(src_dir) and not os.path.exists(dst_dir):
                shutil.copytree(src_dir, dst_dir)
                print(f"\n[WSD] Stable phase ended at step {state.global_step}. Backed up checkpoint to: {dst_dir}")
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        m = kwargs.get("model", self.model)
        print(generate_and_print_sample(m, self.tokenizer, self.device, "数学家希望"))

def data_collator(batch):
    ids = torch.stack([example['input'] for example in batch]).to(torch.int64)
    # print(ids.shape, ids, ids[:,:-1], ids[:,1:])
    return {"input_ids":ids[:,:-1], "labels":ids[:,1:]}


if __name__ == "__main__":
    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA:
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using {device} device.")

    args = {
        "per_device_batch_size": 16,
        "grad_accum_steps": 16,
        "num_epochs": 1,
        "learning_rate": 0.0006
    }

    # 初始化模型
    tokenizer = DeepseekTokenizer.DeepseekTokenizer()
    # config = MyGPTPretrainedModel.MyGPTPretrainedModelConfig(vocab_size=tokenizer.len(), pad_token_id=tokenizer.pad_token_id)
    # model = MyGPTPretrainedModel.MyGPTPretrainedModel(config).to(device, dtype=torch.bfloat16)
    config = MyLlamaPretrainedModel.MyLlamaPretrainedModelConfig(vocab_size=tokenizer.len(), pad_token_id=tokenizer.pad_token_id, layers=24, embedding_dim=512, head_num=8, max_context=1024)
    model = MyLlamaPretrainedModel.MyLlamaPretrainedModel(config)
    # model = MyLlamaPretrainedModel.MyLlamaPretrainedModel.from_pretrained("./results1/checkpoint-47000").to(device, dtype=torch.bfloat16)

    train_dataset = load_from_disk('./TinyStories')
    # train_dataset = train_dataset.to_iterable_dataset()
    # train_dataset = train_dataset.shuffle(123, buffer_size=100)
    # train_dataset = train_dataset.batch(batch_size=16)
    train_dataset = train_dataset.with_format("torch")
    # train_loader = StatefulDataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=False)
    print("Training data loaded")

    # 自动精确计算 Trainer 的总更新步数 (total_steps)
    num_update_steps_per_epoch = math.ceil(
        math.ceil(len(train_dataset) / args["per_device_batch_size"]) / args["grad_accum_steps"]
    )
    total_steps = num_update_steps_per_epoch * args["num_epochs"]
    # 自动分配 WSD 三阶段步数 (大数据集固定 400 步 warmup，小数据集最多占 5% + 82% stable + 15% decay)
    warmup_steps = min(400, max(1, int(total_steps * 0.05)))
    decay_steps = max(1, int(total_steps * 0.15))
    stable_steps = max(0, total_steps - warmup_steps - decay_steps)
    stable_end_step = warmup_steps + stable_steps
    print(f"Total steps: {total_steps} | Warmup: {warmup_steps} | Stable: {stable_steps} (ends at step {stable_end_step}) | Decay: {decay_steps}")

    training_args = TrainingArguments(
        output_dir="./results1",
        num_train_epochs=args["num_epochs"],
        per_device_train_batch_size=args["per_device_batch_size"],
        # ignore_data_skip=True,
        # per_device_eval_batch_size=2,
        gradient_accumulation_steps=args["grad_accum_steps"],
        eval_strategy="no",
        learning_rate=args["learning_rate"],
        # logging_dir="./logs",
        logging_steps=5,
        save_steps=500,
        save_strategy="steps",
        save_total_limit=5,
        report_to="wandb",
        weight_decay=0.1,
        adam_beta2=0.95,
        # seed=123,
        warmup_steps=warmup_steps,
        # lr_scheduler_type="cosine",
        # lr_scheduler_type="cosine_with_min_lr",
        # lr_scheduler_kwargs={"min_lr_rate": 0.1},# 最低保持在 6e-5，避免后期停止更新
        lr_scheduler_type="warmup_stable_decay",
        lr_scheduler_kwargs={
            "num_decay_steps":decay_steps,
            "num_stable_steps":stable_steps,
            "min_lr_ratio":0.1,
            "decay_type": "1-sqrt",
        },
        run_name="test",
        # optim="adamw_bnb_8bit",
        optim="adamw_torch_fused" if USE_CUDA else "adamw_torch",
        dataloader_pin_memory=USE_CUDA,
        torch_compile=True,
        # label_names=['target'],
        remove_unused_columns=False,
        # resume_from_checkpoint=True
        bf16=True,
        tf32=USE_CUDA
    )

    trainer = MyHuggingfaceTrainer.MyHuggingfaceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=tokenized_dataset["validation"],
        callbacks=[LossLoggingCallback(stable_end_step=stable_end_step, model=model, tokenizer=tokenizer, device=device)],
        data_collator=data_collator,
        is_packed_dataset=True,  # 数据集由 g_data_v2 无缝拼接打包，无 pad_token，直接用 batch_size * seq_len 计算有效 token 数
    )

    # trainer.train(resume_from_checkpoint="./results1/checkpoint-72500")  # 开始训练！
    trainer.train()
