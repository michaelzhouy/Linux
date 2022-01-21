## 配置

1. 初始设置

```sh
git config --global user.name "michaelzhouy"
git config --global user.email "15602409303@163.com"

git config --global color.ui auto
```

2. 生成SSH Key, 一直回车, 生成  `id_rsa` 和 `id_rsa.pub` 两个文件, 在`~/.ssh` 路径下
   - 复制公钥 `id_rsa.pub` 到GitHub的Setting/SSH and GPG keys里

```sh
ssh-keygen -t rsa -C "15602409303@163.com"
```

## git命令

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

## 删除分支

```sh
# 删除本地分支
# 先退出分支
git checkout master
# 当一个分支被推送并合并到远程分支后，-d 才会本地删除该分支；如果一个分支还没有被推送或者合并，那么可以使用-D强制删除它
git branch -d dev_zy

# 删除远程分支
git push origin --delete dev_zy

# 不显示远程已被删除的分支
git fetch -p
```

## 拉取开发分支代码

```sh
git clone -b dev_zy github地址
```

