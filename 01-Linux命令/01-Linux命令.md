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
12. 查看文件大小
- ls -lht 查看当前目录下, 每个文件大小, 同时给出当前目录下所有文件大小总和
- ll -lht 查看文件大小
- ls -lh file1 查看file1的文件大小
- du -sh file1 查看file1的文件大小
- du -sh * 查看当前目录下, 所有文件大小总和
13. 解压文件
unzip -o -d . A_CSV.zip 将A_CSV.zip压缩包解压到当前文件夹下