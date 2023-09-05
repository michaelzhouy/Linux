1. 模型缓存路径

```sh
~/.cache/huggingface/hub/
```



1. 不加载模型预训练参数

```python
from transformers import AutoConfig, AutoTokenizer, AutoModel

# 不需将预训练参数上传至obs
config = AutoConfig.from_pretrained('./bert-base-cased/')
model = AutoModel(config=config)

config = AutoConfig.from_pretrained('./bart-large-chinese/')
model = AutoModelForSeq2SeqLM.from_config(config=config)
```

2. Tokenizer_config.json

```
# origin
{"do_lower_case": false,
"do_basic_tokenize": true,
"never_split": null,
"unk_token": "[UNK]",
"sep_token": "[SEP]",
"pad_token": "[PAD]",
"cls_token": "[CLS]",
"mask_token": "[MASK]",
"tokenize_chinese_chars": true,
"strip_accents": null,
"bos_token": "[CLS]",
"eos_token": "[EOS]",
"name_or_path": "/remote-home/yfshao/workdir/code-base/Megatron-LM/init_models_ckpt/bart_zh/base",
"special_tokens_map_file": "vocab/cpt_v3_vocab/special_tokens_map.json",
"tokenizer_file": null
}

# adjust
{"do_lower_case": false,
"do_basic_tokenize": true,
"never_split": null,
"unk_token": "[UNK]",
"sep_token": "[SEP]",
"pad_token": "[PAD]",
"cls_token": "[CLS]",
"mask_token": "2000",
"tokenize_chinese_chars": true,
"strip_accents": null,
"bos_token": "[CLS]",
"eos_token": "[EOS]",
"description_token":"2001",
"diagnosis_token":"2002",
"clinical_token":"2003",
"name_or_path": "/remote-home/yfshao/workdir/code-base/Megatron-LM/init_models_ckpt/bart_zh/base",
"special_tokens_map_file": "vocab/cpt_v3_vocab/special_tokens_map.json",
"tokenizer_file": null
}
```

```
[MASK] 103 1003
[CLS]  101 1001
[EOS]  104 1004
[UNK]  100 1000
[SEP]  102 1002
[PAD]  0   1005
```

0-616