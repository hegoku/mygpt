import argparse
import json
import math
import DeepseekTokenizer
import MyLlamaPretrainedModel
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model loss on a validation dataset.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file (supports a single object or an array of objects)")
    parser.add_argument("--dataset_path", type=str, default=None, help="Validation dataset path on disk")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--batch_size", type=int, default=None, help="Evaluation batch size (default: 8)")

    args = parser.parse_args()

    defaults = {
        "dataset_path": None,
        "checkpoint": None,
        "batch_size": 8,
    }

    raw_configs = [{}]
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        if isinstance(config_data, list):
            if not config_data:
                parser.error("The JSON config array cannot be empty.")
            raw_configs = config_data
        elif isinstance(config_data, dict):
            raw_configs = [config_data]
        else:
            parser.error("--config JSON must be an object or an array of objects.")

    tasks = []
    for idx, item in enumerate(raw_configs):
        if not isinstance(item, dict):
            parser.error(f"Config item at index {idx} must be a JSON object.")

        task_cfg = dict(defaults)
        task_cfg.update(item)

        for key in defaults:
            cli_val = getattr(args, key, None)
            if cli_val is not None:
                task_cfg[key] = cli_val

        task_cfg["batch_size"] = int(task_cfg["batch_size"])

        if not task_cfg["dataset_path"] or not task_cfg["checkpoint"]:
            parser.error(
                f"Task #{idx + 1}: --dataset_path and --checkpoint must be provided either via CLI or in --config JSON file."
            )

        tasks.append(argparse.Namespace(**task_cfg))

    return tasks


def data_collator(batch):
    ids = torch.stack([example["input"] for example in batch])
    return {"input_ids": ids[:, :-1], "labels": ids[:, 1:]}


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    pad_token_id = model.config.pad_token_id

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        valid_tokens = (labels != pad_token_id).sum().item()
        if valid_tokens > 0:
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("nan")
    return avg_loss


if __name__ == "__main__":
    tasks = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using {device} device.")

    tokenizer = DeepseekTokenizer.DeepseekTokenizer()

    current_checkpoint = None
    model = None
    current_dataset_path = None
    eval_dataset = None

    total_tasks = len(tasks)
    for idx, task in enumerate(tasks, start=1):
        print(f"\n[{idx}/{total_tasks}] checkpoint={task.checkpoint} | dataset={task.dataset_path} | batch_size={task.batch_size}")

        if task.checkpoint != current_checkpoint:
            model = MyLlamaPretrainedModel.MyLlamaPretrainedModel.from_pretrained(task.checkpoint).to(
                device, dtype=torch.bfloat16
            )
            current_checkpoint = task.checkpoint

        if task.dataset_path != current_dataset_path:
            eval_dataset = load_from_disk(task.dataset_path).with_format("torch")
            current_dataset_path = task.dataset_path
            print(f"Eval data loaded: {len(eval_dataset)} samples")

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=task.batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )

        val_loss = evaluate(model, eval_loader, device)
        ppl = math.exp(val_loss) if not math.isnan(val_loss) and val_loss < 20 else float("inf")
        print(f"Validation Loss: {val_loss:.4f} | Perplexity: {ppl:.4f}")