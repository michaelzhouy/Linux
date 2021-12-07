## 
1. 克隆远程项目到本地
```sh
git clone git@github.com:michaelzhouy/time-series-forecasting-with-python.git
```
2. 分支
```sh
git branch dev_zy  # 新建开发分支dev_zy
git checkout dev_zy  # 切换开发分支dev_zy
```
3. pull远程master分支代码到本地开发分支
```sh
git pull origin master
```
4. 提交代码
```sh
git add test.txt
git commit -m "add test.txt"
git push origin dev_zy  # 将本地文件(修改或新建)提交至远程开发分支dev_zy
```
5. 删除文件并提交代码
```sh
rm test.txt  # 将本地文件删除(远程master分支已有该文件)
git rm test.txt
git commit -m "remove test.txt"
git push origin dev_zy
```
6. 将开发分支合并到master分支
```sh
git checkout master  # 切换到master分支
git pull origin master  # 拉取远程master分支代码
git merge dev_zy  # 将dev_zy分支合并到master分支
git status  # 查看状态
git push origin master  # 将改动push到远程master上
```