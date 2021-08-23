## Mac笔记本远程Linux的notebook
#### 启动 notebook
1. iTerm进入服务器
```
sh zy_gpu.sh
```
2. 启动 notebook
````
jupyter notebook --allow-root

nohup jupyter notebook --allow-root > jupyter.log 2>&1 &
````
3. 本机在浏览器中输入
```
10.22.20.21:60225
```

### linux配置notebook
1.  jupyter notebook --generate-config --allow-root
2. 修改配置文件
```
vim ~/.jupyter/jupyter_notebook_config.py
# vim命令
i 插入
shift+g 定位最后一行
shift+4 定位到行尾
o 在当前行的下一行添加内容
esc 退出编辑模式
:wq!  保存并退出
:q 退出
```

写入内容：
```
c.NotebookApp.ip='*'
c.NotebookApp.password = u'argonxxx'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 18000 # 可自行指定一个端口, 访问时使用该端口, 后面的那个
c.NotebookApp.notebook_dir = '/data_local/notebook' #  启动路径
c.NotebookApp.token = ''
```
	
密码生成，首先进入ipython
```
from notebook.auth import passwd
passwd()
```

```
tail -200 jupyter.log
```

## ForkLift的使用
1. 协议：SFTP
2. 服务器：10.22.20.21
3. 用户名：root
4. 密码：1
5. 端口：50333

## 修改Host

将以下内容添加至hosts中
```
vim /etc/hosts
```

```
10.21.3.10      hadoop0
10.21.3.11      hadoop1
10.21.3.12      hadoop2
10.21.3.13      hadoop3
10.21.3.14      hadoop4
10.21.3.15      hadoop5
10.21.3.16      hadoop6
10.21.3.17      hadoop7
10.21.3.18      hadoop8
10.21.3.19      hadoop9
10.21.3.20      hadoop10
10.21.3.21      hadoop11
10.21.3.22      hadoop12

10.21.8.5 psd-hadoop005
10.21.8.6 psd-hadoop006
10.21.8.7 psd-hadoop007
10.21.8.8 psd-hadoop008
10.21.8.9 psd-hadoop009
10.21.8.11 psd-hadoop011
10.21.8.13 psd-hadoop013
10.21.3.25 hadoop13
10.21.8.4 psd-hadoop004
10.21.8.5 psd-hadoop005
10.21.8.6 psd-hadoop006
10.21.8.7 psd-hadoop007
10.21.8.8 psd-hadoop008
10.21.8.9 psd-hadoop009
10.21.8.11 psd-hadoop011
10.21.8.13 psd-hadoop013
10.21.8.31 psd-hadoop031
10.21.8.53 psd-hadoop053

10.22.21.31 dm-hadoop001
10.22.21.32 dm-hadoop002
10.22.21.33 dm-hadoop003
10.22.21.34 dm-hadoop004 
10.22.21.35 dm-hadoop005
```