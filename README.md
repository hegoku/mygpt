# Dataset Generation (`g_data_v2.py`)

`g_data_v2.py` tokenizes text records from a Hugging Face dataset (or local dataset directory), filters out records containing `<unk>` tokens, appends the EOS token ID to each record, packs the token stream into fixed-length chunks of `block_size + 1`, and saves the resulting dataset to disk in Arrow format.

## Arguments

| Argument | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--config` | `str` | No | `None` | Path to a JSON configuration file (supports a single object `{...}` or an array of objects `[{...}, ...]`). CLI arguments override values in the config file. |
| `--dataset_path` | `str` | Yes* | `None` | Hugging Face Hub dataset ID (e.g., `roneneldan/TinyStories`) or local path. |
| `--save_path` | `str` | Yes* | `None` | Output directory path where the processed dataset will be saved via `save_to_disk`. |
| `--name`, `--dataset_name` | `str` | No | `None` | Dataset configuration/subset name passed to `load_dataset(..., name=...)`. |
| `--block_size` | `int` | No | `1024` | Context window size. Each saved chunk has length `block_size + 1` (for input/target shifting). |
| `--cache_dir` | `str` | No | `None` | Directory for Hugging Face dataset caching. |
| `--streaming` | `bool` | No | `True` | Whether to load the source dataset in streaming mode (`True` / `False`). |
| `--split` | `str` | No | `train` | Dataset split to process (e.g., `train`, `validation`). |
| `--text_column` | `str` | No | `text` | Name of the column containing text data (e.g., `text`, `content`). |
| `--dtype` | `str` | No | `int64` | Integer type for saved chunks: `int32` (`4 bytes/token`) or `int64` (`8 bytes/token`). Also accepts `32` or `64`. |
| `--max_size` | `str` | No | `None` | Target maximum train output disk size (e.g., `"500MB"`, `"2GB"`, `"1.5G"`). Automatically converts to `max_tokens`. |
| `--max_tokens` | `int` | No | `None` | Stop after packing at least this many training tokens. |
| `--max_samples` | `int` | No | `None` | Maximum number of raw source records (documents) to read from the dataset. |
| `--test_save_path` | `str` | No | `None` | Directory path to save a non-overlapping test split carved out immediately after the train split. |
| `--test_size` | `str` | No | `None` | Target output disk size for the test split (e.g., `"20MB"`, `"100MB"`). Automatically converts to `test_tokens`. |
| `--test_tokens` | `int` | No | `None` | Target number of tokens to pack for the test split after `max_tokens` is reached. |

*\*`--dataset_path` and `--save_path` must be provided either via CLI or inside the `--config` JSON file.*

## Usage Examples

### 1. Command-Line Arguments

Basic usage with defaults (`block_size=1024`, `streaming=True`, `split="train"`):

```bash
python g_data_v2.py \
  --dataset_path roneneldan/TinyStories \
  --save_path /Volumes/TinyStories
```

Specifying all options (including subset `name` and custom `cache_dir`):

```bash
python g_data_v2.py \
  --dataset_path wikimedia/wikipedia \
  --name 20231101.zh \
  --save_path /Volumes/wiki_zh_chunks \
  --block_size 1024 \
  --cache_dir /Volumes/hf_cache \
  --streaming True \
  --split train
```

### 2. Using a JSON Configuration File

The JSON configuration file can be either a **single object** or an **array of objects** (executed sequentially in a loop).

Single task (`config.json`):

```json
{
  "dataset_path": "roneneldan/TinyStories",
  "name": null,
  "save_path": "/Volumes/TinyStories",
  "block_size": 1024,
  "cache_dir": null,
  "streaming": true,
  "split": "train"
}
```

Multiple tasks (`config.json`):

```json
[
  {
    "dataset_path": "roneneldan/TinyStories",
    "save_path": "/Volumes/TinyStories_train",
    "split": "train"
  },
  {
    "dataset_path": "roneneldan/TinyStories",
    "save_path": "/Volumes/TinyStories_val",
    "split": "validation"
  }
]
```

Run with the config file:

```bash
python g_data_v2.py --config config.json
```

Override specific fields from the CLI (CLI arguments take precedence over `config.json`):

```bash
python g_data_v2.py --config config.json --block_size 512 --split validation
```

---

# Model Evaluation (`eval.py`)

`eval.py` loads a trained `MyLlamaPretrainedModel` checkpoint and a pre-tokenized Arrow validation dataset from disk, computes the token-weighted average Cross-Entropy validation loss and perplexity (`exp(loss)`), and prints the results.

## Arguments

| Argument | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--config` | `str` | No | `None` | Path to a JSON configuration file (supports a single object `{...}` or an array of objects `[{...}, ...]`). CLI arguments override values in the config file. |
| `--dataset_path` | `str` | Yes* | `None` | Local directory path to the processed validation dataset (saved via `save_to_disk`). |
| `--checkpoint` | `str` | Yes* | `None` | Path to the model checkpoint directory (loaded via `from_pretrained`). |
| `--batch_size` | `int` | No | `8` | Batch size used during evaluation. |

*\*`--dataset_path` and `--checkpoint` must be provided either via CLI or inside the `--config` JSON file.*

## Usage Examples

### 1. Command-Line Arguments

Basic usage with default batch size (`batch_size=8`):

```bash
python eval.py \
  --dataset_path /Volumes/TinyStories_val \
  --checkpoint ./results/checkpoint-5000
```

Specifying custom batch size:

```bash
python eval.py \
  --dataset_path /Volumes/TinyStories_val \
  --checkpoint ./results/checkpoint-5000 \
  --batch_size 16
```

### 2. Using a JSON Configuration File

The JSON configuration file can be either a **single object** or an **array of objects** (executed sequentially in a loop; unchanged `checkpoint` or `dataset_path` between consecutive tasks are automatically reused without reloading).

Single task (`eval_config.json`):

```json
{
  "dataset_path": "/Volumes/TinyStories_val",
  "checkpoint": "./results/checkpoint-5000",
  "batch_size": 8
}
```

Multiple tasks (`eval_config.json`):

```json
[
  {
    "dataset_path": "/Volumes/TinyStories_val",
    "checkpoint": "./results/checkpoint-5000",
    "batch_size": 8
  },
  {
    "dataset_path": "/Volumes/TinyStories_val",
    "checkpoint": "./results/checkpoint-10000",
    "batch_size": 8
  }
]
```

Run with the config file:

```bash
python eval.py --config eval_config.json
```

Override specific fields from the CLI:

```bash
python eval.py --config eval_config.json --checkpoint ./results/checkpoint-10000 --batch_size 16
```
