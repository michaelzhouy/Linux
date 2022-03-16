https://blog.csdn.net/GouGe_CSDN/article/details/105382920

1. 下载SeiHei字体
2. 找到matplotlib的文件路径

```
import matplotlib
matplotlib.matplotlib_fname()
```

3. 进入上述路径，修改matplotlibrc文件

```
# 以下不注释，并在最后两个后面加SimHei
font.family
font.serif
font.sans-serif

# 解决负号不显示问题，以下不注释，并将True改为False
axes.unicode_minus: True
```

4. 删除matplotlib缓存

```
cd ~/.cache
rm -rf matplotlib/
```

5. 每次导入matplotlib后加一句

```
plt.rcParams['font.sans-serif'] = ['SimHei']
```

