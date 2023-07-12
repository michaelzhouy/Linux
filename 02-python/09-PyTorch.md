1. 设置GPU

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

torch.cuda.empty_cache()

# run.sh
export CUDA_VISIBLE_DEVICES="2"
nohup python3 -u train.py > log 2>&1 &
```

2. 并行

```python
import torch
model = torch.nn.DataParallel(model)
model = model.cuda()
```

3. 参数

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

4. 学习率调整

```python
# 学习率预热
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_training_steps = len(train_loader) * epochs
warm_up_ratio = 0.1
num_warmup_steps = warm_up_ratio * num_training_steps
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# 余弦退火
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2, eta_min=1e-5)
```

5. 线程设定

```python
num_threads = int(cpu_count() / 8)
torch.set_num_threads(num_threads)
```

