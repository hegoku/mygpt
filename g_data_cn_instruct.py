from datasets import load_dataset, Dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer
import DeepseekTokenizer
import torch
import os
import re
# import regex as re
import unicodedata
from datasets import get_dataset_config_names

access_token = ''

tokenizer = DeepseekTokenizer.DeepseekTokenizer()
# tokenizer = AutoTokenizer.from_pretrained("Atom-7B_tokenizer", use_fast=True)

# configs = get_dataset_config_names("Mxode/Chinese-Instruct")

# all_subsets = []

# # Load each subset and add it to the list
# for config in configs:
#     print(f"Loading configuration: {config}")
#     # You might want to specify a split like 'train' or 'validation'
#     # For this example, we'll assume we're loading the 'train' split of each config
#     subset = load_dataset('Mxode/Chinese-Instruct', config, split="train")
#     all_subsets.append(subset)

# # Concatenate all loaded subsets into a single dataset
# merged_dataset = concatenate_datasets(all_subsets)
# merged_dataset.save_to_disk('')

stream_dataset = load_from_disk('')

current_input = []
current_target = []
g1024 = 0
g512 = 0
g256 = 0
g128 = 0
g0 = 0
less_10 = 0
total_tokens = 0
tokens = 0
token_total = 0
special_count = 0
unknown_count = 0
none_count = 0
def stream_and_chunk(dataset, chunk_size=512, stride=128):
    global g1024
    global g512
    global g256
    global g128
    global g0
    global total_tokens
    global tokens
    global token_total
    global less_10
    global special_count
    global unknown_count
    global none_count
    for e in dataset:
        i = 0
        if e['response'] is None or e['prompt'] is None:
            none_count += 1
            continue;
        instruction_text = "<|system|>\nYou are Jack's AI assistant, please answer user's question.\n" + "<|user|>\n" + e['prompt'] + "\n"
        text = instruction_text + "<|assistant|>\n" + e['response'] + "\n" + tokenizer.eos_token
        token_ids = tokenizer.encode(text)
        instruction_ids = tokenizer.encode(instruction_text)
        remaining_size = token_ids.shape[0]
        total_tokens += remaining_size
        unknown = (token_ids==tokenizer.unk_token_id).sum()
        if unknown>0:
            unknown_count +=1
            continue
        if remaining_size>1024:
            g1024 +=1
        elif remaining_size>=512:
            g512 +=1
        elif remaining_size>=256:
            g256 +=1
        elif remaining_size>=128:
            g128 +=1
        else:
            g0 +=1
        tokens += remaining_size
        if remaining_size>1024:
            continue
        while remaining_size>0:
            if (remaining_size<=chunk_size):
                input_chunk = token_ids[i:]
                # target_chunk = token_ids[i + 1: ]
                tmp = torch.full([chunk_size-input_chunk.shape[0]+1], tokenizer.pad_token_id)
                input_chunk = torch.cat((input_chunk, tmp))
                token_total += (input_chunk != tokenizer.pad_token_id).sum().item()
                # tmp = torch.full([chunk_size-target_chunk.shape[0]], tokenizer.pad_token_id)
                # target_chunk = torch.cat((target_chunk, tmp))
            else:
                input_chunk = token_ids[i:i + chunk_size+1]
                token_total += input_chunk.shape[0] - 1
                # target_chunk = token_ids[i + 1: i + chunk_size + 1]
            # yield {"input":tokenizer.decode(input_chunk.detach().clone()), "target":tokenizer.decode(target_chunk.detach().clone())}
            
            # yield {"input":input_chunk, "target":target_chunk}
            # unknown = (input_chunk==tokenizer.unk_token_id).sum()
            # if remaining_size>10:
            #     yield {"input":input_chunk}
            # elif unknown>0:
            #     unknown_count += 1
            # else:
            #     less_10 += 1
            # if unknown>0:
                # unknown_count += 1
            # else:
            yield {"input":input_chunk, 'ins_len':instruction_ids.shape[0]}
            # res.append({"i":input_chunk.detach().clone(), "t":target_chunk.detach().clone()})
            i = i+stride
            remaining_size = remaining_size -stride

        # yield {"id":example['id']}

chunked_dataset = Dataset.from_generator(stream_and_chunk, gen_kwargs={"dataset":stream_dataset, "chunk_size":1024, "stride":1024}, cache_dir='')
# chunked_dataset = Dataset.from_generator(stream_and_chunk, gen_kwargs={"dataset":stream_dataset, "chunk_size":1024, "stride":1024})

chunked_dataset.save_to_disk("")
print(g1024, g512, g256, g128, g0, end="\n")
print(f"total token: {tokens}/{total_tokens}, {token_total}")
print(f"special_count:{special_count} unknown:{unknown_count}")
print(f"none_count:{none_count}")