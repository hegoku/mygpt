from datasets import load_dataset, Dataset, load_from_disk
import DeepseekTokenizer
import torch
import os
import re

tokenizer = DeepseekTokenizer.DeepseekTokenizer()

#1384748
data_path = 'fjcanyue/wikipedia-zh-cn'
stream_dataset = load_dataset(data_path, split='train')
stream_dataset = stream_dataset.batch(batch_size=1000)

current_input = []
current_target = []
g1024 = 0
g512 = 0
g256 = 0
g128 = 0
g0 = 0
def stream_and_chunk(dataset, chunk_size=512, stride=128):
    global g1024
    global g512
    global g256
    global g128
    global g0
    for example in dataset:
        for e in example['text']:
            i = 0
            text = e + tokenizer.eos_token
            token_ids = tokenizer.encode(text)
            remaining_size = token_ids.shape[0]
            if remaining_size<1024:
                continue
            # if remaining_size>=1024:
            #     g1024 +=1
            # elif remaining_size>=512:
            #     g512 +=1
            # elif remaining_size>=256:
            #     g256 +=1
            # elif remaining_size>=128:
            #     g128 +=1
            # else:
            #     g0 +=1
            while remaining_size>0:
                if (remaining_size<=chunk_size):
                    input_chunk = token_ids[i:]
                    target_chunk = token_ids[i + 1: ]
                    tmp = torch.full([chunk_size-input_chunk.shape[0]], tokenizer.pad_token_id)
                    input_chunk = torch.cat((input_chunk, tmp))
                    tmp = torch.full([chunk_size-target_chunk.shape[0]], tokenizer.pad_token_id)
                    target_chunk = torch.cat((target_chunk, tmp))
                else:
                    input_chunk = token_ids[i:i + chunk_size]
                    target_chunk = token_ids[i + 1: i + chunk_size + 1]
                # yield {"input":tokenizer.decode(input_chunk.detach().clone()), "target":tokenizer.decode(target_chunk.detach().clone())}
                yield {"input":input_chunk, "target":target_chunk}
                # res.append({"i":input_chunk.detach().clone(), "t":target_chunk.detach().clone()})
                i = i+stride
                remaining_size = remaining_size -stride

        # yield {"id":example['id']}

chunked_dataset = Dataset.from_generator(stream_and_chunk, gen_kwargs={"dataset":stream_dataset, "chunk_size":1024, "stride":1024})
# for i,b in enumerate(chunked_dataset):
    # aaa = 1
    # print(tokenizer.decode(b['input']))

# print(g1024, g512, g256, g128, g0, end="\n")
chunked_dataset.save_to_disk("train_wiki")

# data_path = './train_wiki'
# stream_dataset = load_from_disk(data_path)
# # print(stream_dataset[1])
# fail= False
# stream_dataset = stream_dataset.to_iterable_dataset()
# stream_dataset = stream_dataset.batch(batch_size=16)
# stream_dataset = stream_dataset.with_format("torch")
# for i, batch in enumerate(stream_dataset):
#     print(batch)
#     break
#     if fail:
#         break
#     for input_b in batch['input']:
#         # a = tokenizer.encode(input_b).tolist()
#         if len(input_b)!=512:
#             print(i,input_b, len(input_b))
#             fail = True
#             break
    # for traget_b in batch['target']:
    #     a = tokenizer.encode(traget_b).tolist()
    #     if len(a)!=512:
    #         print(traget_b)
	# print(i)
	# break