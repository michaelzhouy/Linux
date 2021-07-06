1. pwd: 打印当前路径
2. cd: 切换路径
3. ll: 显示文件夹下的全部文件
4. rm -rf file: 删除文件或文件夹
5. mv file1 file1: 将file1重命名为file2, 如果当前文件夹下也有file2的话, 将会覆盖
6. mv file1 root1: 将file1移动到root1路径下
7. ps -ef: 显示所有进程
8. kill -9 PID: 杀死PID进程
9. tail -200 file1: 显示file1的后200行
10. cp file1 file2: 将file1复制并重命名为file2
11. top: 显示资源使用情况, 按q退出
13. 解压文件
- unzip -o -d . A_CSV.zip 将A_CSV.zip压缩包解压到当前文件夹下
14. 7z 解压文件
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