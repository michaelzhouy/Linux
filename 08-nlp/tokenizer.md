```python
import torch
from transformers import BertTokenizer

model_name = 'bert-base-uncased'

# 通过词典导入分词器
tokenizer = BertTokenizer.from_pretrained(model_name)
sentence = "Hello, my son is laughing."
# 分词
tokenizer.tokenize(sentence)

# tokenizer.encode_plus(sentence, add_special_tokens=False)不包含[cls], [sep]这些
print(tokenizer.encode(sentence)) # tokenizer.encode()只返回input_ids，列表类型

# tokenizer(sentence)与tokenizer.encode_plus(sentence, add_special_tokens=True)一样，返回input_ids、token_type_ids、attention_mask，input_ids为单词在词典中的编码，token_type_ids区分两个句子的编码（上句全为0，下句全为1），attention_mask指定对哪些词进行self-Attention操作
print(tokenizer(sentence))
print(tokenizer.encode_plus(sentence, add_special_tokens=True))
print(tokenizer.encode_plus(sentence, add_special_tokens=False))
```

