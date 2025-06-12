import torch
import Tokenizer
import MyDataset
import Attention
import MyGPT
import DeepseekTokenizer

# tokenizer = Tokenizer.Tokenizer()
tokenizer = DeepseekTokenizer.DeepseekTokenizer()
text = "这里我们演示在我们网站里下载。"
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))
pass

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(tokenizer.len(), 20)

dataloader = MyDataset.create_dataloader_v1(text, tokenizer, max_length=6, stride=6, shuffle=False)
token_embedding = embedding_layer(ids)
print(token_embedding.shape)

batch = torch.stack((token_embedding, token_embedding), dim=0)
# attention = Attention.CausalAttention(token_embedding.shape[0], token_embedding.shape[1], d_q=2, d_v=4, dropout=0.5)
# print(attention(batch))
# head_att = Attention.MultiHeadAttentionWrapper(token_embedding.shape[0], token_embedding.shape[1], d_q=2, d_v=4, dropout=0.5, head_num=2)
# print(head_att(token_embedding))
# print(attention.state_dict)

# t = MyGPT.TransformerBlock(token_embedding.shape[0], token_embedding.shape[1], d_q=2, d_v=4, dropout=0.5, head_num=2)
# print(t(token_embedding))

gpt = MyGPT.MyGPT(tokenizer=tokenizer, layer=1 , max_context=token_embedding.shape[0], embedding_dim=token_embedding.shape[1], d_q=2, d_v=4, dropout=0.0, head_num=1)
# a = gpt(ids)
# print(a.shape)

new_idx = MyGPT.generate_text(gpt, torch.tensor(ids).unsqueeze(0), 10, 6)
print(new_idx)
print(tokenizer.decode(new_idx[0]))
