## 查看文件大小

[磁盘爆满分析](https://blog.csdn.net/mr_wanter/article/details/112515814)

1. 查看当前目录总大小
```sh
df -h
du -h --max-depth=1 ./
du -h --max-depth=1
```
2. 查看当前目录下, 每个文件大小, 同时给出当前目录下所有文件大小总和
```sh
ls -lht
```
3. 查看file1的文件大小
```sh
ls -lh file1
```
4. 查看file1的文件大小
```sh
du -sh file1
```
5. 查看当前目录下, 所有文件大小总和
```sh
du -sh *
```