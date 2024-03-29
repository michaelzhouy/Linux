## 对抗训练

### FGM

```python
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}
    
    def attack(self, epsilon=0.1, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
    
    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

fgm = FGM(model)
for batch_input, batch_label in data:
    loss = model(batch_input, batch_label)
    loss.backward()  

    # adversarial training
    fgm.attack() 
    loss_adv = model(batch_input, batch_label)
    loss_adv.backward() 
    fgm.restore()

    optimizer.step()
    model.zero_grad()
```

### PGD

```python
class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='emb.', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]


pgd = PGD(model)
K = 3
for batch_input, batch_label in data:
    # 正常训练
    loss = model(batch_input, batch_label)
    loss.backward() # 反向传播，得到正常的grad
    pgd.backup_grad()
    # 对抗训练
    for t in range(K):
        pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
        if t != K-1:
            model.zero_grad()
        else:
            pgd.restore_grad()
        loss_adv = model(batch_input, batch_label)
        loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
    pgd.restore() # 恢复embedding参数
    # 梯度下降，更新参数
    optimizer.step()
    model.zero_grad()
```

### AWP

```python
class AWP:
    def __init__(self, model, optimizer, adv_param="weight", adv_lr=1, adv_eps=0.2, start_epoch=0, adv_step=1, scaler=None):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.scaler = scaler

    def attack_backward(self, x, y, attention_mask, epoch):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save()
        for i in range(self.adv_step):
            self._attack_step()
            with torch.cuda.amp.autocast():
                adv_loss, tr_logits = self.model(input_ids=x, attention_mask=attention_mask, labels=y)
                adv_loss = adv_loss.mean()
            self.optimizer.zero_grad()
            self.scaler.scale(adv_loss).backward()

        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1])
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self, ):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


def train():
    scaler = torch.cuda.amp.GradScaler()
    awp = AWP(model, optimizer, adv_lr=1, adv_eps=1e-3, start_epoch=1, scaler=scaler)
    step = 0
    for e in range(epochs):
        for idx, batch in enumerate(train_loader):
        	with torch.cuda.amp.autocast():
            	loss, tr_logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            	loss = loss.mean()

        	optimizer.zero_grad()
        	scaler.scale(loss).backward()
        	if e > 0:
            	awp.attack_backward(input_ids, labels, attention_mask, e)

        	# gradient clipping
        	torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)
        	scaler.step(optimizer)
        	scaler.update()
        	scheduler.step()
```

## 模型

### SWA

```python
# https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/#how-to-use-swa-in-pytorch
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

loader, optimizer, model, loss_fn = ...
swa_model = AveragedModel(model)
scheduler = CosineAnnealingLR(optimizer, T_max=100)
swa_start = 5
swa_scheduler = SWALR(optimizer, swa_lr=0.05)

for epoch in range(100):
      for input, target in loader:
          optimizer.zero_grad()
          loss_fn(model(input), target).backward()
          optimizer.step()
      if epoch > swa_start:
          swa_model.update_parameters(model)
          swa_scheduler.step()
      else:
          scheduler.step()

# Update bn statistics for the swa_model at the end
torch.optim.swa_utils.update_bn(loader, swa_model)
# Use swa_model to make predictions on test data 
preds = swa_model(test_input)
```

### EMA

```python
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
 
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
 
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
 
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
 
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
 
# 初始化
ema = EMA(model, 0.999)
ema.register()
 
# 训练过程中，更新完参数后，同步update shadow weights
def train():
    optimizer.step()
    ema.update()
 
# eval前，apply shadow weights；eval之后，恢复原来模型的参数
def evaluate():
    ema.apply_shadow()
    # evaluate
    ema.restore()
```

### Multi Dropout

```python
class MarkdownModel(nn.Module):
    def __init__(self, model_path):
        super(MarkdownModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.fc = nn.Linear(768, 1)

    def forward(self, ids, mask):
        x = self.model(ids, mask)[0]
        # x = self.dropout1(x)
        # x = self.fc(x[:, 0, :])
        pred1 = self.fc(self.dropout1(x)[:, 0, :])
        pred2 = self.fc(self.dropout2(x)[:, 0, :])
        pred3 = self.fc(self.dropout3(x)[:, 0, :])
        pred4 = self.fc(self.dropout4(x)[:, 0, :])
        pred5 = self.fc(self.dropout5(x)[:, 0, :])
        pred = (pred1 + pred2 + pred3 + pred4 + pred5) / 5
        return pred
```

### 梯度裁剪

```python
scaler = GradScaler()
for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()
        with autocast():
            output = model(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()
        # 梯度裁剪之前没有恢复
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()
```

### DDP梯度累加

```python
for epoch in range(epoches):
    for j, data in enumerate(train_loader):
        # 前accumulation_steps - 1个step，不进行梯度同步，累积梯度。
        if accumulation_count % accumulation_steps != 0:
            with model.no_sync():
                loss = model(data)
                loss = loss / accumulation_steps
                loss.backward()
        else:
            loss = model(data)
            loss = loss / accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            model_optimizer.step()
            if model_scheduler is not None:
                model_scheduler.step()
            model_optimizer.zero_grad()
         accumulation_count += 1


# 优雅的写法（兼容单卡和DDP模式）
from contextlib import nullcontext
# 如果python版本小于3.7，则使用下面这个：
# from contextlib import suppress as nullcontext

if local_rank != -1:
    model = DDP(model)

accumulation_count = 0
optimizer.zero_grad()
for epoch in range(epoches):
    for i, data in enumerate(train_loader):
        # 只在DDP模式下，轮数不是accumulation_steps整数倍的时候使用no_sync
        mcontext = model.no_sync if local_rank != -1 and accumulation_count % accumulation_steps != 0 else nullcontext
        with mcontext():
            loss = model(data)
            loss = loss / accumulation_steps
            loss.backward()
        # 轮数为accumulation_steps整数倍的时候，传播梯度，并更新参数
        if accumulation_count % accumulation_steps == 0:
            optimizer.step()
            if model_scheduler is not None:
                model_scheduler.step()
            optimizer.zero_grad()
        accumulation_count += 1
```

## Loss

### PolyLoss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')


def poly1_cross_entropy_torch(logits, labels, class_number=3, epsilon=1.0):
    poly1 = torch.sum(F.one_hot(labels, class_number).float() * F.softmax(logits), dim=-1)
    ce_loss = F.cross_entropy(torch.tensor(logits), torch.tensor(labels), reduction='none')
    poly1_ce_loss = ce_loss + epsilon * (1 - poly1)
    return poly1_ce_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3):
        super(FocalLoss, self).__init__()
        self.alpha = torch.zeros(num_classes)
        self.alpha[0] += alpha
        self.alpha[1:] += (1 - alpha)
        self.gamma = gamma

    def forward(self, logits, labels):
        logits = logits.view(-1, logits.size(-1))
        self.alpha = self.alpha.to(logits.device)
        logits_logsoft = F.log_softmax(logits, dim=1)
        logits_softmax = torch.exp(logits_logsoft)
        logits_softmax = logits_softmax.gather(1, labels.view(-1, 1))
        logits_logsoft = logits_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - logits_softmax), self.gamma), logits_logsoft)
        loss = torch.mul(self.alpha, loss.t())[0, :]
        return loss


def poly1_focal_loss_torch(logits, labels, alpha=0.25, gamma=2, num_classes=3, epsilon=1.0):
    focal_loss_func = FocalLoss(alpha, gamma, num_classes)
    focal_loss = focal_loss_func(logits, labels)

    p = torch.sigmoid(logits)
    labels = torch.nn.functional.one_hot(labels, num_classes)
    labels = torch.tensor(labels, dtype=torch.float32)
    poly1 = labels * p + (1 - labels) * (1 - p)
    poly1_focal_loss = focal_loss + torch.mean(epsilon * torch.pow(1 - poly1, 2 + 1), dim=-1)
    return poly1_focal_loss


if __name__ == '__main__':
    logits = [[2, 0.5, 1],
              [0.1, 1, 3]]
    labels = [1, 2]
    print("PyTorch loss result:")
    ce_loss = F.cross_entropy(torch.tensor(logits), torch.tensor(labels), reduction='none')
    print("torch cross_entropy:", ce_loss)

    poly1_ce_loss = poly1_cross_entropy_torch(torch.tensor(logits), torch.tensor(labels), class_number=3, epsilon=1.0)
    print("torch poly1_cross_entropy:", poly1_ce_loss)

    focal_loss_func = FocalLoss(alpha=0.25, gamma=2, num_classes=3)
    fc_loss = focal_loss_func(torch.tensor(logits), torch.tensor(labels))
    print("torch focal_loss:", fc_loss)

    poly1_fc_loss = poly1_focal_loss_torch(torch.tensor(logits), torch.tensor(labels), alpha=0.25, gamma=2,
                                           num_classes=3, epsilon=1.0)
    print("torch poly1_focal_loss:", poly1_fc_loss)
```

### FocalLoss

```python
class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""
    def __init__(self, gamma=2, weight=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        log_pt = torch.log_softmax(input, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = torch.nn.functional.nll_loss(log_pt, target, self.weight, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss
```

### LabelSmoothing

```python
# https://github.com/wangleiofficial/label-smoothing-pytorch
def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float=0.1, reduction='mean', ignore_index=-100):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction, ignore_index=self.ignore_index)
        return linear_combination(loss / n, nll, self.epsilon)
```

### Sparsemax

```python
class Sparsemax(nn.Module):
    """Sparsemax loss"""
    def __init__(self, k_sparse=1000):
        super(Sparsemax, self).__init__()
        self.k_sparse = k_sparse
        
    def forward(self, preds, labels):
        """
        Args:
            preds (torch.Tensor):  [batch_size, number_of_logits]
            labels (torch.Tensor): [batch_size] index, not ont-hot
        Returns:
            torch.Tensor
        """
        preds = preds.reshape(preds.size(0), -1) # [batch_size, -1]
        topk = preds.topk(self.k_sparse, dim=1)[0] # [batch_size, k_sparse]
        
        # log(sum(exp(topk)))
        pos_loss = torch.logsumexp(topk, dim=1)
        # s_t
        neg_loss = torch.gather(preds, 1, labels[:, None].expand(-1, preds.size(1)))[:, 0]
        return (pos_loss - neg_loss).sum()
```

## 对比学习

### R-Drop

```python
import torch.nn.functional as F

# define your task model, which outputs the classifier logits
model = TaskModel()

def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss

# keep dropout and forward twice
logits = model(x)
logits2 = model(x)
# cross entropy loss for classifier
ce_loss = (cross_entropy_loss(logits, label) + cross_entropy_loss(logits2, label)) / 2
kl_loss = compute_kl_loss(logits, logits2)
# carefully choose hyper-parameters, alpha=1-10
loss = ce_loss + alpha * kl_loss
```

## 扰动

### NoisyTune

```python
# https://zhuanlan.zhihu.com/p/523865674
model = MT5ForConditionalGeneration.from_pretrained(model_path)
noise_lambda = 0.2
for name, para in model.named_parameters():
    model.state_dict()[name][:] += (torch.rand(para.size()) - 0.5) * noise_lambda * torch.std(para)
model.to(device)
```



### model soup

```python
def soup_two_models(model, second_model):
    souped_model = copy.deepcopy(model)
    for param in souped_model.named_parameters():
        name = param[0]
        param[1].data = (model.state_dict()[name] + second_model.state_dict()[name]) / 2

    return souped_model
```



