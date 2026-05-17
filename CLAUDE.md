# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

## 项目概述

LLMCompiler 是一个用于 LLM 高效并行函数调用的框架。它将问题分解为可并行执行的任务，识别任务之间的依赖关系，并据此编排执行流程。该项目支持 OpenAI 模型，以及通过 vLLM 或 Friendli 端点提供的开源模型。

## 命令

### 环境配置
```bash
uv venv --python 3.10              # 创建虚拟环境（指定 Python 3.10）
source .venv/bin/activate          # 激活环境 (Linux/macOS)
# 或 .venv\Scripts\activate       # Windows
uv pip install -r requirements.txt # 安装依赖
```

### uv 常用指令

#### 虚拟环境管理
```bash
uv venv                          # 创建 .venv 虚拟环境
uv venv --python 3.10            # 指定 Python 版本创建
source .venv/bin/activate        # 激活环境 (Linux/macOS)
```

#### 包管理
```bash
uv pip install -r requirements.txt   # 从 requirements.txt 安装依赖
uv pip install <package>             # 安装单个包
uv pip install <package>==<version>  # 安装指定版本
uv pip uninstall <package>           # 卸载包
uv pip list                          # 列出已安装的包
uv pip freeze > requirements.txt     # 导出当前依赖
```

### 运行基准测试
```bash
# 首先设置 API 密钥
export OPENAI_API_KEY="sk-xxx"

# 在基准测试上运行 LLMCompiler
python run_llm_compiler.py --benchmark_name {hotpotqa|movie|parallelqa} --store results.json [--stream] [--logging]

# 使用 vLLM 端点运行
python run_llm_compiler.py --model_type vllm --benchmark_name {benchmark} --store results.json --model_name {model-name} --vllm_port {port}

# 使用 Friendli 端点运行
FRIENDLI_TOKEN="xxx" python run_llm_compiler.py --model_type friendli --benchmark_name {benchmark} --store results.json --model_name {model-name}

# 运行 ReAct 基线进行对比
python run_llm_compiler.py --benchmark_name {benchmark} --store results.json --react
```

### 评估结果
```bash
python evaluate_results.py --file results.json [--detail]
```

### 主要命令行参数
- `--benchmark_name`: 必填。可选值为 `hotpotqa`、`movie` 或 `parallelqa`
- `--store`: 必填。保存结果的路径（JSON 格式）
- `--stream`: 推荐。启用流式输出以降低延迟
- `--react`: 使用 ReAct 智能体替代 LLMCompiler（用于基线对比）
- `--model_type`: `openai`（默认）、`vllm`、`azure` 或 `friendli`
- `--N`: 限制样本数量
- `--do_benchmark`: 收集详细的运行时统计信息

## 架构

### 核心组件 (`src/llm_compiler/`)

1. **LLMCompiler** (`llm_compiler.py`): 主编排器，协调规划和执行。接收工具、LLM 实例和提示词；处理计划-执行-合并循环，支持可选的重规划。

2. **Planner** (`planner.py`): 通过调用 LLM 生成执行计划。使用流式输出逐步生成任务。创建包含工具描述和并行执行指南的系统提示词。

3. **TaskFetchingUnit** (`task_fetching_unit.py`): 管理带依赖追踪的任务执行。并行执行独立任务，阻塞依赖任务直到其依赖项完成。使用 `$id` 语法引用先前任务的输出。

4. **Output Parser** (`output_parser.py`): 将 LLM 输出解析为 `Task` 对象。处理计划格式：`1. tool_name(args)`、`2. other_tool($1)` 等。

### 关键数据结构

- **Task**: 包含 idx（索引）、name（名称）、tool function（工具函数）、args（参数）、dependencies（依赖）、observation（观察结果）和 stringify_rule（字符串化规则）。`is_join` 标志标记最终的合并操作。

### 配置系统 (`configs/`)

每个基准测试都有一个专用目录，包含：
- `configs.py`: 模型默认值和提示词引用（GPT 和 LLaMA 风格模型使用不同的提示词）
- `tools.py`: 工具定义，包含描述和字符串化规则
- `gpt_prompts.py` / `llama_prompts.py`: 用于规划和输出生成的上下文示例

### 添加自定义基准测试

1. 在 `configs/{benchmark_name}/` 下创建目录
2. 在 `tools.py` 中定义工具，包含清晰的描述
3. 创建包含上下文示例的提示词文件
4. 在 `run_llm_compiler.py` 中添加配置加载逻辑

### 依赖语法

任务使用 `$id` 或 `${id}` 引用先前任务的输出：
```
1. search("query")
2. process($1)  # 使用任务 1 的输出
3. search("other")  # 独立任务，可与任务 1 并行执行
```

### 环境变量

- `OPENAI_API_KEY`: OpenAI 模型必需
- `OPENAI_BASE_URL`: OpenAI API 基础 URL（用于兼容 API，如 DeepSeek）
- `OPENAI_MODEL`: 默认模型名称（可通过 `--model_name` 覆盖）
- `HTTP_PROXY` / `HTTPS_PROXY`: HTTP 代理（用于 Wikipedia 访问）
- `AZURE_ENDPOINT`、`AZURE_OPENAI_API_VERSION`、`AZURE_DEPLOYMENT_NAME`、`AZURE_OPENAI_API_KEY`: 用于 Azure OpenAI
- `FRIENDLI_TOKEN`: 用于 Friendli 端点

### 模型配置

项目支持通过 `.env` 文件配置模型：

```bash
# .env 示例（DeepSeek API 兼容模式）
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-v4-flash
HTTP_PROXY=http://127.0.0.1:10808
HTTPS_PROXY=http://127.0.0.1:10808
```

**重要**: 运行时需通过 `--model_name` 指定模型，否则使用配置文件中的默认值：
```bash
uv run python run_llm_compiler.py --benchmark_name hotpotqa --store results.json --stream --model_name deepseek-v4-flash
```

### 工具配置

每个基准测试的工具定义：

| 基准测试 | 工具 | 说明 |
|----------|------|------|
| hotpotqa | `search` | 仅 Wikipedia 搜索 |
| movie | `search`, `math` | 搜索 + 数学计算（需 model_name） |
| parallelqa | `search`, `math` | 搜索 + 数学计算（需 model_name） |

**注意**: movie 和 parallelqa 的工具需要 `model_name` 参数来初始化 LLMMathChain。

### 测试数据验证

验证 DAG 生成准确性和参数传递正确性：

```bash
# 1. 收集数据
bash scripts/collect_test_data.sh 30

# 2. 验证并生成报告
uv run python validate_test_data.py

# 3. 查看报告
cat results/validation_report.md
```

验证内容包括：
- 循环依赖检测（Kahn 算法）
- 参数引用有效性（`$id` 和 `${id}`）
- 依赖一致性检查
- 并行度和依赖深度统计
