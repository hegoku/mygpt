from torch.utils.data import Dataset, DataLoader
import torch
import zhconv

class MyDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt)
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(input_chunk.detach().clone())
            self.target_ids.append(target_chunk.detach().clone())

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader_v1(txt, tokenizer, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Create dataset
    dataset = MyDataset3(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

class MyDataset2(Dataset):
    def __init__(self, txt_d, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        for txt in txt_d:
            i = 0
            # txt['text'] = txt['text'].replace("\n\n", tokenizer.eos_token+tokenizer.bos_token)
            token_ids = tokenizer.encode(zhconv.convert(txt['text'],"zh-hans"))
            remaining_size = token_ids.shape[0]
            while remaining_size>0:
                if (remaining_size<=max_length):
                    input_chunk = token_ids[i:]
                    target_chunk = token_ids[i + 1: ]
                    tmp = torch.full([max_length-input_chunk.shape[0]], tokenizer.pad_token_id)
                    input_chunk = torch.cat((input_chunk, tmp))
                    tmp = torch.full([max_length-target_chunk.shape[0]], tokenizer.pad_token_id)
                    target_chunk = torch.cat((target_chunk, tmp))
                else:
                    input_chunk = token_ids[i:i + max_length]
                    target_chunk = token_ids[i + 1: i + max_length + 1]
                self.input_ids.append(input_chunk.detach().clone())
                self.target_ids.append(target_chunk.detach().clone())
                i = i+stride
                remaining_size = remaining_size -stride
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

class MyDataset3(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        i = 0
        # txt['text'] = txt['text'].replace("\n\n", tokenizer.eos_token+tokenizer.bos_token)
        token_ids = tokenizer.encode(txt+tokenizer.eos_token)
        remaining_size = token_ids.shape[0]
        while remaining_size>0:
            if (remaining_size<=max_length):
                input_chunk = token_ids[i:]
                target_chunk = token_ids[i + 1: ]
                tmp = torch.full([max_length-input_chunk.shape[0]], tokenizer.pad_token_id)
                input_chunk = torch.cat((input_chunk, tmp))
                tmp = torch.full([max_length-target_chunk.shape[0]], tokenizer.pad_token_id)
                target_chunk = torch.cat((target_chunk, tmp))
            else:
                input_chunk = token_ids[i:i + max_length]
                target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(input_chunk.detach().clone())
            self.target_ids.append(target_chunk.detach().clone())
            i = i+stride
            remaining_size = remaining_size -stride
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]