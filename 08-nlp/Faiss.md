```python
from transformers import BertTokenizer, T5Model
import torch
 
MODEL_NAME = 'imxly/t5-copy'
print(f'Loading {MODEL_NAME} Model...')
 
# 加载模型和tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = T5Model.from_pretrained(MODEL_NAME)
 
# 输入文本并进行tokenizer
text = ['Hello world!', 'Hello python!']
inputs = tokenizer(text, return_tensors='pt', padding=True)


output = model.encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], return_dict=True)
# print(f'output.shape = {output.shape}')
pooled_sentence = output.last_hidden_state # shape is [batch_size, seq_len, hidden_size]
print(f'pooled_sentence.shape = {pooled_sentence.shape}')
# pooled_sentence will represent the embeddings for each word in the sentence
# you need to sum/average the pooled_sentence
pooled_sentence = torch.mean(pooled_sentence, dim=1)
# print(f'pooled_sentence.shape = {pooled_sentence.shape}')
 
# 得到n_sample*512的句向量
print('pooled_sentence.shape', pooled_sentence.shape)
print(pooled_sentence)
 
# 输出：
# pooled_sentence.shape torch.Size([2, 512])
# tensor([[ 0.0123,  0.0010,  0.0202,  ..., -0.0176,  0.0122, -0.1353],
#         [ 0.0854,  0.0613, -0.0568,  ...,  0.0230, -0.0131, -0.2288]],
#        grad_fn=<MeanBackward1>)
```

https://huggingface.co/learn/nlp-course/chapter5/6?fw=pt