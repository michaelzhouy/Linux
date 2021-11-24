https://blog.csdn.net/qq_32447301/article/details/79387649

## docker常用命命令

1. 登录
```sh
sudo docker login registry.gz.cvte.cn
```

### 容器
1. 查看容器
```sh
docker ps -aq # 查看所有容器ID
docker ps -a # 查看所有容器以及其他信息
```
2. 停止容器(这样才能够删除其中的镜像)
```sh
docker stop <container id> # 停止容器
docker stop $(docker ps -a -q) # 停止所有的容器
docker stop $(docker ps -aq) # 停止所有的容器
```
3. 对容器的操作
```sh
docker stop <container id> # 停止容器
docker kill <container id> # kill容器
docker start <container id> # 启动容器
docker restart <container id> # 重启容器
docker attach id # 重新进去
```
4. 如果想要删除所有container(容器)的话再加一个指令
```sh
docker rm $(docker ps -a -q)
docker rm $(docker ps -aq)
docker container prune # 删除所有停止的容器
docker image prune --force --all # 删除所有不使用的镜像
docker image prune -f -a
```
5. 退出
```sh
exit # 停止docker
ctrl + p + q # 不停止docker退出
```

### 镜像
1. 删除镜像
```sh
docker images # 查看所有镜像以及其他信息
docker images -q # 查看所有镜像
docker rmi <image id> # 删除指定镜像
docker rmi $(docker images -q) # 删除所有镜像
docker rmi -f $(docker images -q) # 强制删除所有镜像
```
2. 运行
```sh
docker images # 查看镜像, 获取镜像id
docker run -it 45f95ffc1c49 bash # 进入docker
docker run -d -p 7533:7533 e98dfefa76fc python3 ./app/interface.py # 后台运行, 指定端口
docker run -d --name consume -it e98dfefa76fc bash # 后台运行, 指定容器名字
docker run -d --name consume e98dfefa76fc # 这种是指定好dockerfile后运行
```
3. docker进入容器, 查看配置文件
```sh
docker exec: 在运行的容器中执行命令
        -d: 分离模式, 后台运行
        -i: 即使没有附加也保持STDIN(标准输入)打开, 以交互模式运行容器, 通常与 -t 同时使用
        -t: 为容器重新分配一个伪输入终端，通常与 -i 同时使用；
docker exec -it  f94d2c317477 /bin/bash
```
4. 修改配置, 退出容器
```sh
1. 如果要正常退出不关闭容器, 请按Ctrl+P+Q进行退出容器
2. 如果使用exit退出, 那么在退出之后会关闭容器, 可以使用下面的流程进行恢复
使用docker restart命令重启容器
使用docker attach命令进入容器
```
5. 挂载类目, 映射id, 启动多个命令
```sh
docker run -d -p 8081:8080 -v /etc/localtime:/etc/localtime:ro --add-host=dp-master001.gz.cvte.cn:10.21.25.161 --add-host=dp-master001:10.21.25.161 c7ce71ad3d33 sh -c './usr/chenjw/apache-tomcat-8.5.47/bin/startup.sh && tail -f ./usr/chenjw/apache-tomcat-8.5.47/logs/catalina.out'
```
6. 想要删除untagged images, 也就是那些id为的image的话可以用
```sh
docker rmi $(docker images | grep "^<none>" | awk "{print $3}")
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