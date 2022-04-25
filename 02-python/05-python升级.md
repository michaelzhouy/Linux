1. 安装python3.7

```sh
apt update
apt install software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt install python3.7

python -V
```

2. pip安装

```sh
python3.7 -m pip install --upgrade pip # pip可能挂了

# pip安装
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

python3.7 get-pip.py --force-reinstall


# 配置pip链接
https://www.jianshu.com/p/16790a512ad7

~/.pip/pip.conf

[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host=mirrors.aliyun.com
```

3. 修改软连接

```sh
which python
whereis python

# 删除python软链接
$ rm -rf /usr/bin/python

ln -s 源文件 目标文件
```

4. pip安装

```sh
pip uninstall urllib3
pip install --no-cache-dir -U urllib3
pip uninstall chardet
pip install --no-cache-dir -U chardet
```



