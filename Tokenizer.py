import jieba
import pandas as pd
import torch

class Tokenizer:
    bos_token: str = '<|startoftext|>'
    eos_toekn: str = '<|endoftext|>'
    pad_token: str = '<|pad|>'
    unk_token: str = '<|unk|>'
    dict = []
    rev_dict = {}

    def __init__(self, dict_path=None):
        self.dict.append(self.bos_token)
        self.dict.append(self.eos_toekn)
        self.dict.append(self.pad_token)
        self.dict.append(self.unk_token)
        self.dict.append("\n")
        self.dict.append("!")
        self.dict.append(".")
        self.dict.append(",")
        self.dict.append("/")
        self.dict.append("'")
        self.dict.append('"')
        self.dict.append('?')
        self.dict.append('~')
        self.dict.append('@')
        self.dict.append('#')
        self.dict.append('$')
        self.dict.append('%')
        self.dict.append('^')
        self.dict.append('&')
        self.dict.append('*')
        self.dict.append('()')
        self.dict.append(')')
        self.dict.append('_')
        self.dict.append('+')
        self.dict.append('-')
        self.dict.append('=')
        self.dict.append('--')
        self.dict.append('`')
        self.dict.append(']')
        self.dict.append('[')
        self.dict.append('{')
        self.dict.append('}')
        self.dict.append('|')
        self.dict.append('\\')
        self.dict.append(':')
        self.dict.append(';')
        self.dict.append('<')
        self.dict.append('>')
        self.dict.append('	')
        self.dict.append('～')
        self.dict.append('·')
        self.dict.append('！')
        self.dict.append('@')
        self.dict.append('#')
        self.dict.append('¥')
        self.dict.append('%')
        self.dict.append('……')
        self.dict.append('（')
        self.dict.append('）')
        self.dict.append('【')
        self.dict.append('】')
        self.dict.append('「')
        self.dict.append('」')
        self.dict.append('；')
        self.dict.append('‘')
        self.dict.append('：')
        self.dict.append('“')
        self.dict.append('”')
        self.dict.append('，')
        self.dict.append('。')
        self.dict.append('/')
        self.dict.append('《')
        self.dict.append('》')
        self.dict.append('？')
        self.dict.append('、')
        self.dict.append('｜')
        self.rev_dict = {i:s for s,i in enumerate(self.dict)}

        if dict_path!=None:
            jieba.load_userdict(dict_path) 
            df = pd.read_csv(dict_path, delimiter=" ", header=None, usecols=[0])
        for i, text in (enumerate(df[0])):
            if self.rev_dict.get(text)==None:
                self.dict.append(text)
                self.rev_dict[text] = len(self.dict)-1

    def print(self):
        for i, item in enumerate(self.dict[0:10]):
            print(item, i)

    def encode(self, text:str):
        fenci = jieba.cut(text)
        res = []
        # res.append(self.rev_dict[self.bos_token])
        for i, text in (enumerate(fenci)):
            if self.rev_dict.get(text)==None:
                res.append(self.rev_dict[self.unk_token])
            else:
                res.append(self.rev_dict[text])
        # res.append(self.rev_dict[self.eos_toekn])
        return torch.tensor(res)

    def decode(self, ids) -> str:
        res = []
        for id in ids:
            res.append(self.dict[id])
        return " ".join(res)

    def len(self):
        return len(self.dict)
