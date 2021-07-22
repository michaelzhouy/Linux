1. 挂起执行
```
nohup python -u train.py >> nolog 2>&1 &
```
2. 显示日志
```
tail -f nolog
```