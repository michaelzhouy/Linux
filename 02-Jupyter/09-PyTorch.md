1. 设置GPU

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
```



1. 并行

```python
import torch
model = torch.nn.DataParallel(model)
model = model.cuda()
```

2. 参数

```sh
nohup python3 -u train.py \
--data_dir='../data/' \
--model_dir='./model/model_pth/' \
--epochs=20 \
--model='swin_base_patch4_window7_224' \
--batch_size=128 \
--input_size=224 \
--LR=1e-3 \
--num_workers=8 \
--cuda='1,2,3,4' >> swin_base_224_log 2>&1 &
```

3. 
