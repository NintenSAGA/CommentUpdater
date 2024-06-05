# LLMCup - LLM Comment Updater

## Build & Run

运行前需要安装 [Ollama](https://ollama.com/)

依赖通过 [Poetry](https://python-poetry.org/) 进行管理，首次执行前需要执行下述指令以安装依赖：

```bash
poetry install
```

可以通过以下方式执行：

```bash
# 1 - 直接运行
poetry run python3 ./src/main.py
# 2 - Makefile
make run
```

## 文件结构

```
.
├── config      # 配置文件
├── data        # 输入数据
├── jupyter     # 一些数据处理 jupyter
├── logs        # 日志
├── modelfiles  # Ollama 模型文件
├── prompts     # 提示词
├── result      # 输出结果
└── src         # 代码源文件
```

