1. Jupyter Notebook内核Python环境位置

```sh
ipython kernelspec list
```

2. 进入上述路径下的python文件夹下的kernel.json文件

```sh
{
 "argv": [
  "python",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "Python 3 (ipykernel)",
 "language": "python",
 "metadata": {
  "debugger": true
 }

 {
    "argv": [
     "/usr/bin/python3.7",
     "-m",
     "ipykernel_launcher",
     "-f",
     "{connection_file}"
    ],
    "display_name": "Python 3",
    "language": "python"
   }
```

