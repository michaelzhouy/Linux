1. 生成秘钥

```sh
# 生成gitlab秘钥: 在~/.ssh/路径下会生成id_rsa和id_rsa.pub, 将id_rsa.pub配置到gitlab的SSH秘钥中
ssh-keygen -t rsa -C "公司邮箱地址"

# 生成github秘钥: 将github_rsa.pub配置到github的SSH秘钥中
ssh-keygen -t rsa -C "github地址" -f ~/.ssh/github_rsa

ssh-keygen -t rsa -C "15602409303@163.com" -f ~/.ssh/gitlab_rsa
```

2. 配置config

```sh
# gitlab
Host gitlab
    HostName gitlab.xx.cn # 公司gitlab地址
    User git
    IdentityFile ~/.ssh/id_rsa
# github
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/github_rsa
```

3. 测试连接

```sh
ssh -T git@gitlab
ssh -T git@github.com
```

