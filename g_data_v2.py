import argparse
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
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset path or HuggingFace repo ID (e.g. roneneldan/TinyStories)")
    parser.add_argument("--name", "--dataset_name", dest="name", type=str, default=None, help="Dataset configuration/subset name for load_dataset (default: None)")
    parser.add_argument("--save_path", type=str, required=True, help="Directory path to save the processed dataset")
    parser.add_argument("--block_size", type=int, default=1024, help="Context block size (default: 1024, chunk length will be block_size + 1)")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for HuggingFace datasets (default: None)")
    parser.add_argument("--streaming", type=str2bool, nargs="?", const=True, default=True, help="Whether to stream the dataset (default: True)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to load (default: train)")
    return parser.parse_args()


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
    args = parse_args()
    cache_dir = args.cache_dir if args.cache_dir else None
    dataset_name = args.name if args.name else None

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
