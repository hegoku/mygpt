from transformers import AutoTokenizer

class DeepseekTokenizer():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek_tokenizer")

    def encode(self, text:str):
        return self.tokenizer.encode(text, add_special_tokens=False, padding_side='right', return_tensors='pt')[0]
    
    def decode(self, ids) -> str:
        return self.tokenizer.decode(ids)
    
    def len(self):
        return len(self.tokenizer)