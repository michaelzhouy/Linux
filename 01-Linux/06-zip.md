1. unzip解压
- unzip -o -d . A_CSV.zip 将A_CSV.zip压缩包解压到当前文件夹下
2. 7z 解压文件
    1. 安装

```sh
sudo apt-get install p7zip-full

7z x filename.7z filename
```

- 7z x manager.7z -r -o/.
    - x 代表解压缩文件, 并且是按原始目录解压(还有个参数 e 也是解压缩文件, 但其会将所有文件都解压到根下, 而不是自己原有的文件夹下)
    - manager.7z 是压缩文件, 这里大家要换成自己的. 如果不在当前目录下要带上完整的目录
    - -r 表示递归所有的子文件夹
    - -o 是指定解压到的目录, 这里大家要注意-o后是没有空格的直接接目录
- 7z a -t7z -r manager.7z /home/manager/*
    - a 代表添加文件／文件夹到压缩包
    - -t 是指定压缩类型, 一般定为7z
    - -r 表示递归所有的子文件夹
    - manager.7z 是压缩好后的压缩包名
    - /home/manager/* 是要压缩的目录, ＊是表示该目录下所有的文件

3. Tar.gz

```sh
tar -zxvf fenci.py.tar.gz -C sample/
```

