import argparse
import json
from datasets import load_dataset, Dataset
import DeepseekTokenizer
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize and pack dataset into fixed-size chunks.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file (supports a single object or an array of objects)")
    parser.add_argument("--dataset_path", type=str, default=None, help="Dataset path or HuggingFace repo ID (e.g. roneneldan/TinyStories)")
    parser.add_argument("--name", "--dataset_name", dest="name", type=str, default=None, help="Dataset configuration/subset name for load_dataset (default: None)")
    parser.add_argument("--save_path", type=str, default=None, help="Directory path to save the processed dataset")
    parser.add_argument("--block_size", type=int, default=None, help="Context block size (default: 1024, chunk length will be block_size + 1)")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for HuggingFace datasets (default: None)")
    parser.add_argument("--streaming", type=str2bool, nargs="?", const=True, default=None, help="Whether to stream the dataset (default: True)")
    parser.add_argument("--split", type=str, default=None, help="Dataset split to load (default: train)")

    args = parser.parse_args()

    defaults = {
        "dataset_path": None,
        "name": None,
        "save_path": None,
        "block_size": 1024,
        "cache_dir": None,
        "streaming": True,
        "split": "train",
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

        item_copy = dict(item)
        if "dataset_name" in item_copy and "name" not in item_copy:
            item_copy["name"] = item_copy.pop("dataset_name")

        task_cfg = dict(defaults)
        task_cfg.update(item_copy)

        for key in defaults:
            cli_val = getattr(args, key, None)
            if cli_val is not None:
                task_cfg[key] = cli_val

        task_cfg["streaming"] = (
            str2bool(task_cfg["streaming"])
            if not isinstance(task_cfg["streaming"], bool)
            else task_cfg["streaming"]
        )
        task_cfg["block_size"] = int(task_cfg["block_size"])

        if not task_cfg["dataset_path"] or not task_cfg["save_path"]:
            parser.error(
                f"Task #{idx + 1}: --dataset_path and --save_path must be provided either via CLI or in --config JSON file."
            )

        tasks.append(argparse.Namespace(**task_cfg))

    return tasks


tokenizer = DeepseekTokenizer.DeepseekTokenizer()

# --- 2. 创建一个流，负责 Tokenization 和 UNK 过滤 ---
unknown_count = 0
def tokenized_stream_generator(dataset):
    global unknown_count
    print("Starting stream: Tokenizing and filtering UNK tokens...")
    for record in dataset:
        text = record['text']
        if not text:
            continue

        # Tokenize 单个记录并转为 Python int 列表，然后直接追加 eos_token_id
        tokenized_output = tokenizer.encode(text).tolist()
        
        # 检测 UNK token
        if tokenizer.unk_token_id in tokenized_output:
            unknown_count += 1
            # 如果发现 UNK，则直接跳过这个记录，不进行 yield
            continue
        
        tokenized_output.append(tokenizer.eos_token_id)

        # 将合格的、已经 tokenized 的记录 yield 出去
        yield tokenized_output

# --- 3. 创建一个流，负责拼接和分块 ---
token_total = 0
def chunk_stream_generator(dataset, block_size=1024):
    global token_total
    buffer = []
    print("\nStarting stream: Packing tokenized records into chunks...")
    for token_list in tokenized_stream_generator(dataset):
        buffer.extend(token_list)
        
        while len(buffer) >= (block_size + 1):
            chunk = torch.tensor(buffer[:block_size + 1])
            buffer = buffer[block_size:]
            token_total += chunk.shape[0] - 1
            yield {"input": chunk}


if __name__ == "__main__":
    tasks = parse_args()
    total_tasks = len(tasks)

    for idx, args in enumerate(tasks, start=1):
        unknown_count = 0
        token_total = 0
        cache_dir = args.cache_dir if args.cache_dir else None
        dataset_name = args.name if args.name else None

        print(f"\n[{idx}/{total_tasks}] dataset={args.dataset_path} (name={dataset_name}, split={args.split}) -> {args.save_path}")

        stream_dataset = load_dataset(
            args.dataset_path,
            name=dataset_name,
            split=args.split,
            streaming=args.streaming,
            cache_dir=cache_dir,
        )

        chunked_dataset = Dataset.from_generator(
            chunk_stream_generator,
            gen_kwargs={"dataset": stream_dataset, "block_size": args.block_size},
            cache_dir=cache_dir,
        )
        chunked_dataset.save_to_disk(args.save_path)
        print(f"Total token: {token_total}")
        print(f"Total unknown token: {unknown_count}")
