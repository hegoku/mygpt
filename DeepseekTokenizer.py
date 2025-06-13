from transformers import AutoTokenizer

class DeepseekTokenizer():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek_tokenizer")
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token

    def encode(self, text:str):
        return self.tokenizer.encode(text, padding_side='right', return_tensors='pt')[0]
    
    def decode(self, ids) -> str:
        return self.tokenizer.decode(ids)
    
    def len(self):
        return len(self.tokenizer)