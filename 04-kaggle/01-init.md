## 

1. 安装kaggle

```sh
pip3 install kaggle
```

2. 下载 `kaggle.json` 到 `~/.kaggle/kaggle.json` ，并赋权

```sh
chmod 600 ~/.kaggle/kaggle.json
```

3. 上传本地文件

```sh
# 安装nano
apt-get install -y nano

# 初始化metadata file，并通过nano命令修改id和title参数
kaggle datasets init -p folder/

# Ctrl+O保存，Ctrl+X退出，如果退出前没有保存会提示Y或N
nano mydataset/dataset-metadata.json

# -r tar子目录也上传
kaggle datasets create -p folder/ -r tar

# dataset version
kaggle datasets version -p folder/ -m "Updated data" -r tar
```

