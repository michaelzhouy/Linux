1. dtale

```python
import dtale
import pandas as pd
df = pd.read_csv('./data/titanic.csv')
d = dtale.show(df)
d.open_browser()
```



1. pandas_profiling

- 安装

```sh
pip3 install ipywidgets==7.6.5
```

 - 使用

```python
import pandas as pd
from pandas_profiling import ProfileReport
df = pd.DataFrame()
profile = ProfileReport(df, title="Pandas Profiling Report")
profile
```

3. sweetviz

```python
import sweetviz as sv 
sweetviz_report = sv.analyze(df)
sweetviz_report.show_html() 
```

4. autoviz

```python
from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()

sep = ';'
dft = AV.AutoViz(filename="",sep=sep, depVar='Pclass', dfte=df, header=0, verbose=2, 
                 lowess=False, chart_format='png', max_rows_analyzed=150000, max_cols_analyzed=30)
```

