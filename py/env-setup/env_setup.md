### 1. 部署 Anaconda 到你当前的操作系统
### 2. 创建一个 AI_Python_for_Intermediates 的 Python 环境，该环境 使用 Python3.11 作为运行环境
```
conda create -n AI_Python_for_Intermediates python=3.11
```


### 3. 配置 pip 镜像加速第三方库的安装过程(MacOs)
(1)创建或编辑 pip 配置文件： 在用户主目录下创建或编辑 .pip/pip.conf 文件。如果 .pip 目录不存在，可以先创建该目录
```
mkdir -p ~/.pip
```
(2)添加镜像源： 在 pip.conf 文件中添加以下内容，以使用国内的镜像源（例如清华大学的镜像源）
```
echo "[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple" > ~/.pip/pip.conf
```

### 4. 安装 openai、gradio 第三方库
```
pip install openai gradio 
```

