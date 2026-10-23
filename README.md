# Dataset Generation (`g_data_v2.py`)

`g_data_v2.py` tokenizes text records from a Hugging Face dataset (or local dataset directory), filters out records containing `<unk>` tokens, appends the EOS token ID to each record, packs the token stream into fixed-length chunks of `block_size + 1`, and saves the resulting dataset to disk in Arrow format.

## Arguments

| Argument | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--config` | `str` | No | `None` | Path to a JSON configuration file. CLI arguments override values in the config file. |
| `--dataset_path` | `str` | Yes* | `None` | Hugging Face Hub dataset ID (e.g., `roneneldan/TinyStories`) or local path. |
| `--save_path` | `str` | Yes* | `None` | Output directory path where the processed dataset will be saved via `save_to_disk`. |
| `--name`, `--dataset_name` | `str` | No | `None` | Dataset configuration/subset name passed to `load_dataset(..., name=...)`. |
| `--block_size` | `int` | No | `1024` | Context window size. Each saved chunk has length `block_size + 1` (for input/target shifting). |
| `--cache_dir` | `str` | No | `None` | Directory for Hugging Face dataset caching. |
| `--streaming` | `bool` | No | `True` | Whether to load the source dataset in streaming mode (`True` / `False`). |
| `--split` | `str` | No | `train` | Dataset split to process (e.g., `train`, `validation`). |

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

Create a `config.json` file:

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

Run with the config file:

```bash
python g_data_v2.py --config config.json
```

Override specific fields from the CLI (CLI arguments take precedence over `config.json`):

```bash
python g_data_v2.py --config config.json --block_size 512 --split validation
```
