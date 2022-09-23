1. hf下载模型到指定目录下

```python
!pip install huggingface_hub
from huggingface_hub import hf_hub_download
# ["config.json", "pytorch_model.bin", "vocab.txt"]
hf_hub_download("hfl/chinese-bert-wwm-ext", filename="vocab.txt", cache_dir="./")

from huggingface_hub import snapshot_download
snapshot_download(repo_id='hfl/chinese-bert-wwm-ext', allow_regex=['config.json', 'vocab.txt', 'pytorch_model.bin'], cache_dir='./')
```

2. git下载

```sh
sudo apt install git-lfs
git lfs install
git clone https://huggingface.co/hfl/chinese-bert-wwm-ext
```
