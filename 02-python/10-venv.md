1. 安装virtualenv

```
pip3 install virtualenv
virtualenv --version
```

2. 进入项目路径，创建venv文件夹，创建虚拟环境并激活

```
cd ..
mkdir venv

virtualenv venv
source ./venv/bin/activate

# 退出虚拟环境
deactivate
```

3. 在虚拟环境下安装依赖包

```
pip3 install -r requirements.txt

pip install -i https://mirrors.aliyun.com/pypi/simple pytorch-lightning==1.4.9 ----no-deps

# 安装fairscale出错，加上后面的参数
pip install -i https://mirrors.aliyun.com/pypi/simple fairscale --no-build-isolation
```

4. 删除虚拟环境，直接删除目录

```
rm -rf venv
```