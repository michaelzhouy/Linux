## Notebook

1. 导入
```python
# 输出多个结果
from IPython.core.interactiveshell import InteractiveShell 
InteractiveShell.ast_node_interactivity = 'all'

# 过滤警告
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

# DataFrame显示所有列
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
```
2. sub

```python
# 保存sub
df[['sid', 'label']].to_csv('../sub/sub_{}_1.csv'.format(time.strftime('%Y%m%d')), index=False)
```

3. 代码快捷键

- Ctrl + Enter - 运行当前的cell
- Shift + Enter - 运行当前的cell并进入下一个cell
- Space: 向下滑动cell
- a - 在上面插入一个cell
- b - 在下面插入一个cell
- m - 将cell转化为markdown
- y - 将cell转化为code
- x - 裁剪选中的cell
- c - 对选中的cell进行拷贝
- v - 对选中的cell进行黏贴
- z - 撤销cell的删除

