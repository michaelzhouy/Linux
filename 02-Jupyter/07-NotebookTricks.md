1. 输出多个结果
```python
from IPython.core.interactiveshell import InteractiveShell 
InteractiveShell.ast_node_interactivity = 'all'
```
2. 过滤警告

```python
import warnings
warnings.filterwarnings('ignore')
```



2. 代码快捷键
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

