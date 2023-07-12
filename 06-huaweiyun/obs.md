## obs下载

```
# 下载
wget https://obs-community.obs.cn-north-1.myhuaweicloud.com/obsutil/current/obsutil_linux_amd64.tar.gz
# 解压
tar -xzvf obsutil_linux_amd64.tar.gz
# 进入文件夹做赋权
chmod 755 obsutil
```

## 配置

1. 创建 ak sk 参考链接 https://support.huaweicloud.com/eu/tg-modelarts/modelarts_15_0004.html

```
# 检查网络是否通畅
ping obs.cn-north-4.myhuaweicloud.com

# 配置ak+sk，obs.cn-north-4.myhuaweicloud.com（北京4）
./obsutil config -i=ak -k=sk -e=obs.cn-north-4.myhuaweicloud.com

ak=1DVMZ5JR1RR0UIEZIT6Q
sk=ofiegLVKbINhs8LujmwldeUdRQ4DeVdgap7W4R7z

# 检查连通性
./obsutil ls -s
```

## 命令

https://support.huaweicloud.com/utiltg-obs/obs_11_0018.html

```
# 将本地文件上传至obs
./obsutil cp ../04-huaweicloud/07-kingmedcc1/05-bart-base/baseline_model.pt obs://kingmedcc-2023-turbo/model/baseline_model.pt

./obsutil cp ../04-huaweicloud/07-kingmedcc1/03-cpt-base-maxlen/customize_service.py obs://kingmedcc-2023-turbo/model/customize_service.py

# 将本地文件夹上传至obs
./obsutil cp ../04-huaweicloud/bert-models/cpt-base/ obs://kingmedcc-2023-turbo/model/ -r -f

# 将obs文件下载至本地
./obsutil cp obs://turbo-models/Randeng-T5-784M-QA-Chinese .\ -r -f
```



