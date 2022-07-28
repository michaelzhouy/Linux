1. 修改docker时区

```
apt-get update
apt-get install tzdata
```

2. 选择 Asia/Shanghai

https://blog.csdn.net/qq_32447301/article/details/79387649

## docker常用命令

1. 登录
```sh
sudo docker login registry.gz.cvte.cn
```

2. 镜像

```sh
docker images # 显示所有镜像以及其他信息
docker images -q # 显示所有镜像
docker rmi <image_id> # 删除指定镜像(需先将容器删除)
docker rmi $(docker images -q) # 删除所有镜像
docker rmi -f $(docker images -q) # -f 强制删除所有镜像
docker image prune # 删除所有悬空的镜像
docker image prune --force --all # 删除所有不使用的镜像
docker image prune -f -a
```
3. 容器

```sh
# 查看容器
docker ps # 查看成功的容器
docker ps -aq # 查看所有容器ID
docker ps -a # 查看所有容器以及其他信息(包括失败的容器)

# 停止容器
docker stop <container id> # 停止容器(这样才能够删除其中的镜像)
docker stop $(docker ps -a -q) # 停止所有的容器
docker stop $(docker ps -aq) # 停止所有的容器

docker kill <container id> # kill容器

docker start <container id> # 启动容器
docker restart <container id> # 重启容器
docker attach <container id> # 重新进去

# 删除容器
docker rm <container id> # 删除指定容器
docker rm $(docker ps -a -q)
docker rm $(docker ps -aq)
docker container prune # 删除所有停止的容器

# 退出容器
ctrl + p + q # 退出容器, 不关闭容器
exit # 退出容器并关闭
使用docker restart命令重启容器
使用docker attach命令进入容器
```

4. 运行容器

```sh
docker run -it <image_id> bash # 启动一个bash终端, 允许用户进行交互
docker run -d --name ckg_scrapy <image_id> sh run.sh # 后台运行, 指定容器名字
docker run -d -p 7533:7533 <image_id> python3 ./app/interface.py # 后台运行, 指定端口

docker ps
docker logs -tf <container id> # 查看容器日志
```
3. docker进入容器, 查看配置文件
```sh
docker exec: 在运行的容器中执行命令
        -d: 分离模式, 后台运行
        -i: 即使没有附加也保持STDIN(标准输入)打开, 以交互模式运行容器, 通常与 -t 同时使用
        -t: 为容器重新分配一个伪输入终端，通常与 -i 同时使用；
docker exec -it <image_id> /bin/bash
```

6. 挂载类目, 映射id, 启动多个命令
```sh
docker run -d -p 8081:8080 -v /etc/localtime:/etc/localtime:ro --add-host=dp-master001.gz.cvte.cn:10.21.25.161 --add-host=dp-master001:10.21.25.161 c7ce71ad3d33 sh -c './usr/chenjw/apache-tomcat-8.5.47/bin/startup.sh && tail -f ./usr/chenjw/apache-tomcat-8.5.47/logs/catalina.out'
```

1.  从容器到宿主机复制
```sh
docker cp tomcat：/webapps/js/text.js /home/admin
docker cp 容器名:  容器路径       宿主机路径
```
2. 从宿主机到容器复制
```sh
docker cp /home/admin/text.js tomcat：/webapps/js
docker cp 宿主路径中文件      容器名  容器路径  
```

## push自己的镜像

1. 安装docker(Mac安装)，后续步骤都是在iterm命令行中运行
2. 拉取基础镜像

```sh
docker pull registry.gz.cvte.cn/cvte-ai/pytorch:cuda10-torch1.4
```

3. 进入docker, 安装所需要的包

```sh
docker run -tid --name xxx registry.gz.cvte.cn/cvte-ai/pytorch:cuda10-torch1.4 /bin/bash
# example
docker run -tid --name sc registry.gz.cvte.cn/cvte-ai/pytorch:cuda10-torch1.4 /bin/bash

# 挂载本地目录（安装chrome driver）
docker run -tid --name <自定义容器名称> -v /Users/z/Downloads/chromedriver:/usr/local/bin <image id> bash

# 报错：404 Not Found [IP: 185.125.190.36 80]，先执行
apt update

pip3 install xx

# 退出容器
exit
```

4. push镜像

```sh
docker commit -m="add zbar package" -a="xzp" <container id> registry.gz.cvte.cn/dm/pytorch-py3:sc # 或者docker tag <container id> registry.gz.cvte.cn/dm/pytorch-py3:sc

docker login https://registry.gz.cvte.cn/dm/ # 登陆habor, 输入账号和密码
docker push registry.gz.cvte.cn/dm/pytorch-py3:sc
```

