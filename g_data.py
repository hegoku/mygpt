from datasets import load_dataset, Dataset, load_from_disk
import DeepseekTokenizer
import torch
import os
import zhconv

tokenizer = DeepseekTokenizer.DeepseekTokenizer()

#1384748
# data_path = 'wikimedia/wikipedia'
# stream_dataset = load_dataset(data_path, "20231101.zh", split='train[0:1107798]')
# # stream_dataset = load_dataset(data_path, "20231101.zh", split='train[1107798:]')
# stream_dataset = stream_dataset.to_iterable_dataset()
# # stream_dataset = stream_dataset.shuffle(42, buffer_size=1000)
# stream_dataset = stream_dataset.batch(batch_size=1000)

# current_input = []
# current_target = []
# def stream_and_chunk(dataset, chunk_size=512):
#     for example in dataset:
#         # res = []
#         for e in example['text']:
#             i = 0
#             text = e + tokenizer.eos_token
#             token_ids = tokenizer.encode(zhconv.convert(text,"zh-hans"))
#             remaining_size = token_ids.shape[0]
#             while remaining_size>0:
#                 if (remaining_size<=chunk_size):
#                     input_chunk = token_ids[i:]
#                     target_chunk = token_ids[i + 1: ]
#                     tmp = torch.full([chunk_size-input_chunk.shape[0]], tokenizer.eos_token_id)
#                     input_chunk = torch.cat((input_chunk, tmp))
#                     tmp = torch.full([chunk_size-target_chunk.shape[0]], tokenizer.eos_token_id)
#                     target_chunk = torch.cat((target_chunk, tmp))
#                 else:
#                     input_chunk = token_ids[i:i + chunk_size]
#                     target_chunk = token_ids[i + 1: i + chunk_size + 1]
#                 # yield {"input":tokenizer.decode(input_chunk.detach().clone()), "target":tokenizer.decode(target_chunk.detach().clone())}
#                 yield {"input":input_chunk.detach().clone(), "target":target_chunk.detach().clone()}
#                 # res.append({"i":input_chunk.detach().clone(), "t":target_chunk.detach().clone()})
#                 i = i+chunk_size
#                 remaining_size = remaining_size -chunk_size

#         # yield {"id":example['id'], "text": res}

# chunked_dataset = Dataset.from_generator(stream_and_chunk, gen_kwargs={"dataset":stream_dataset})

# for i in chunked_dataset:
	# print(i)
# chunked_dataset.save_to_disk("train_wiki")

data_path = './train_wiki'
stream_dataset = load_from_disk(data_path)
# print(stream_dataset[1])
fail= False
stream_dataset = stream_dataset.to_iterable_dataset()
stream_dataset = stream_dataset.batch(batch_size=16)
stream_dataset = stream_dataset.with_format("torch")
for i, batch in enumerate(stream_dataset):
    print(batch)
    break
    if fail:
        break
    for input_b in batch['input']:
        # a = tokenizer.encode(input_b).tolist()
        if len(input_b)!=512:
            print(i,input_b, len(input_b))
            fail = True
            break
    # for traget_b in batch['target']:
    #     a = tokenizer.encode(traget_b).tolist()
    #     if len(a)!=512:
    #         print(traget_b)
	# print(i)
	# break