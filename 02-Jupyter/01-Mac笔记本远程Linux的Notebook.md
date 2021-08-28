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
1. jupyter notebook --generate-config --allow-root
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