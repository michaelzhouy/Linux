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

### medcc1

```sh
./obsutil cp ../04-huawei/01-medcc1/03-bart-base/01-bart-base-maxlen512/customize_service.py obs://medcc1/model/customize_service.py

./obsutil cp ../04-huawei/03-medcc1/01-bart-base-maxlen512-lr2e-4/customize_service.py obs://medcc1/model/bart-base-state/customize_service.py

./obsutil cp ../04-huawei/01-medcc1/03-bart-base-maxlen512-seed773/baseline_model.pt obs://medcc1/model/bart-base-seed773/baseline_model.pt
```





### medcc2

```sh
./obsutil cp ../04-huawei/bert-models/t5-copy/ obs://medcc2/model/ -r -f

./obsutil cp ../04-huawei/bert-models/t5-copy-med-qa/ obs://medcc2/model/ -r -f

./obsutil cp ../04-huawei/bert-models/t5-copy-summary/ obs://medcc2/model/ -r -f

./obsutil cp ../04-huawei/04-medcc2/01-t5-med-epoch10-bs10-seed369/baseline_model.pt obs://medcc2/model/t5-med-seed369/baseline_model.pt

./obsutil cp ../04-huawei/04-medcc2/02-t5-copy-epoch10-bs10-seed5693/baseline_model.pt obs://medcc2/model/t5-copy-5693/baseline_model.pt

./obsutil cp ../04-huawei/04-medcc2/customize_service.py obs://medcc2/model/customize_service.py
```



```
# 将本地文件上传至obs
./obsutil cp ../04-huawei/07-kingmedcc1/03-cpt-base/customize_service.py obs://medcc1/model/customize_service.py

./obsutil cp ../04-huawei/07-kingmedcc1/03-cpt-base/baseline_model.pt  obs://medcc1/model/baseline_model.pt

./obsutil cp ../04-huawei/bert-models/cpt-base/ obs://medcc1/model/ -r -f


./obsutil cp ../04-huawei/07-kingmedcc1/00-pretrain/01-cpt-base-qa-epoch5/customize_service.py obs://kingmedcc-2023-turbo/model/customize_service.py

./obsutil cp ../04-huawei/07-kingmedcc1/03-cpt-base-maxlen/customize_service.py obs://kingmedcc-2023-turbo/model/customize_service.py

# 将本地文件夹上传至obs
./obsutil cp ../04-huawei/bert-models/cpt-base/ obs://kingmedcc-2023-turbo/model/ -r -f

./obsutil cp ../04-huawei/bert-models/t5-copy/ obs://kingmedcc2-2023-turbo/model/ -r -f

# 将obs文件下载至本地
./obsutil cp obs://turbo-models/Randeng-T5-784M-QA-Chinese .\ -r -f
```



```
./obsutil cp ../04-huawei/08-kingmedcc2/01-t5-small/baseline_model.pt obs://kingmedcc2-2023-turbo/model/baseline_model.pt

./obsutil cp ../04-huawei/12-2-ciderd-t5-copy/01-t5-copy-epoch10-bs20/customize_service_test.py obs://kingmedcc2-2023-turbo/model/customize_service.py

# 将本地文件夹上传至obs
./obsutil cp ../04-huawei/bert-models/cpt-base/ obs://kingmedcc2-2023-turbo/model/ -r -f
```



```sh
./obsutil cp ../04-huawei/来杯凉白开-0372-top12.zip obs://kingmedcc-2023-ttt/competition1/来杯凉白开-0372-top12.zip

./obsutil cp ../04-huawei/CVTEDMer-12717-top6.zip obs://kingmedcc-2023-ttt/competition2/CVTEDMer-12717-top6.zip
```



