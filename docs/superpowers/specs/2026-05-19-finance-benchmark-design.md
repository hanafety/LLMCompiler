# Finance Benchmark Design

## 目标

新增 `finance` 基准测试，使用财经领域复杂场景测试 LLMCompiler 的依赖生成和参数生成精确度。区别于现有 hotpotqa/movie/parallelqa，本基准测试侧重 DAG 结构正确性验证而非端到端答案正确性。

## 数据集

### 结构

每条测试数据包含：

```json
{
    "id": "1",
    "question": "比较苹果公司和微软公司2023年的净利润率，哪家更高？",
    "answer": "微软",
    "complexity": "deep",
    "expected_dag": {
        "tasks": [
            {"idx": 1, "tool": "search", "args": ["Apple Inc."]},
            {"idx": 2, "tool": "search", "args": ["Microsoft"]},
            {"idx": 3, "tool": "math", "args": ["Apple net profit margin in 2023", ["$1"]]},
            {"idx": 4, "tool": "math", "args": ["Microsoft net profit margin in 2023", ["$2"]]},
            {"idx": 5, "tool": "join", "args": []}
        ],
        "dependencies": {
            "3": [1],
            "4": [2],
            "5": [3, 4]
        }
    }
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 唯一标识 |
| `question` | string | 财经问题（中文） |
| `answer` | string | 期望答案 |
| `complexity` | string | `"shallow"` / `"medium"` / `"deep"` |
| `expected_dag.tasks` | array | 期望任务列表，含 idx/tool/args |
| `expected_dag.dependencies` | object | 任务依赖映射，key 为依赖方任务 idx，value 为它所依赖的前驱任务 idx 列表 |

### 复杂度分类

- **shallow**：1 层依赖深度，2-3 个并行分支
- **medium**：2 层依赖深度，3-4 个分支
- **deep**：3+ 层依赖深度，或复杂参数引用（如 context 引用多个前驱任务输出）

### 复杂度分布

- deep: 60-70%（16-18 条）
- medium: 20-30%（5-8 条）
- shallow: 10%（2-3 条）

### 场景覆盖

- 财务报表分析：公司营收、利润率、负债率对比
- 投资组合计算：回报率、风险评估
- 宏观经济指标：GDP、通胀率、利率计算
- 综合场景：跨维度交叉分析

### 规模

20-30 条精选数据，重在质量而非数量。

## 工具

复用 parallelqa 的工具配置：

- `search(entity)` — Wikipedia 搜索，获取公司/经济数据
- `math(problem, context)` — 数学计算，支持引用前驱任务输出

不添加额外财经专用工具，保持与现有基准测试一致。

## 评估指标

### 任务级别

| 指标 | 计算方式 | 说明 |
|------|----------|------|
| Task Recall | 正确生成任务数 / 期望任务总数 | 是否遗漏必要任务 |
| Task Precision | 正确生成任务数 / 生成任务总数 | 是否产生冗余任务 |
| Tool Accuracy | 工具名称匹配的任务数 / 总任务数 | 工具选择是否正确 |

### 依赖级别

| 指标 | 计算方式 | 说明 |
|------|----------|------|
| Dependency Recall | 正确识别的依赖边数 / 期望依赖边总数 | 是否遗漏依赖 |
| Dependency Precision | 正确识别的依赖边数 / 生成依赖边总数 | 是否多余依赖 |
| DAG Isomorphism | DAG 同构的样本数 / 总样本数 | 整体结构是否一致 |

### 参数级别

| 指标 | 计算方式 | 说明 |
|------|----------|------|
| Arg Reference Accuracy | 参数引用完全匹配的任务数 / 需引用的任务总数 | `$id` 或 `${id}` 引用是否正确 |
| Arg Value Accuracy | 非引用参数值匹配的任务数 / 此类任务总数 | 如搜索词是否正确 |

### 报告格式

```
=== Finance Benchmark Evaluation Report ===
Total Samples: 25

[Complexity Breakdown]
  shallow (3):  DAG Match 100%, Dep Recall 95%, Arg Acc 92%
  medium (6):   DAG Match 83%,  Dep Recall 88%, Arg Acc 85%
  deep (16):    DAG Match 69%,  Dep Recall 75%, Arg Acc 68%

[Overall Metrics]
  Task Recall:    0.89
  Task Precision: 0.92
  Dep Recall:     0.81
  Dep Precision:  0.87
  Arg Ref Acc:    0.76
  DAG Isomorphism: 0.72
```

## 评估实现

### DAG 比对算法

1. 规范化任务表示（忽略 idx 顺序）
2. 基于任务内容做最优匹配（匈牙利算法）
3. 在任务匹配基础上比对依赖边
4. 检查 DAG 同构性（networkx）

**关键处理：**
- `search("Apple")` 和 `search("Apple Inc.")` 视为等价（字符串相似度 > 0.8）
- `math` 任务的 `context` 参数做集合比对，忽略顺序
- DAG 同构使用 networkx 图同构算法

### 运行流程

```
run_llm_compiler.py --benchmark_name finance --store results.json
                              ↓
                        生成 results.json（含 generated_dag）
                              ↓
evaluate_finance.py --file results.json --detail
                              ↓
                        输出评估报告
```

### results.json 新增字段

```json
{
    "id": "1",
    "question": "...",
    "answer": "微软",
    "generated_dag": {
        "tasks": [...],
        "dependencies": {...}
    },
    "final_answer": "微软",
    "success": true
}
```

## 提示词设计

### PLANNER_PROMPT

核心要点：
- 角色定位为金融分析助手
- 明确强调并行搜索和依赖语法
- 示例覆盖：双公司并行搜索、多参数引用、深度依赖链
- 中文财经问题示例
- 每个示例以 `join()<END_OF_PLAN>` 结尾

### OUTPUT_PROMPT

核心要点：
- 要求简洁答案（单词、数字或公司名）
- 使用 `Finish(answer)` 格式
- 提供财经场景示例

## 配置集成

### configs/finance/configs.py

导入 gpt/llama 提示词，定义 CONFIGS 字典，`max_replans: 1`。

### configs/finance/tools.py

直接复用 parallelqa 的 `tools` 和 `generate_tools`。

### run_llm_compiler.py 修改

1. `--benchmark_name` choices 添加 `"finance"`
2. `get_dataset()` 添加 finance 分支
3. `get_tools()` 添加 finance 分支（复用 parallelqa 工具）
4. `get_configs()` 添加 finance 分支

## 文件清单

```
新增文件:
├── configs/finance/
│   ├── configs.py
│   ├── tools.py
│   ├── gpt_prompts.py
│   └── llama_prompts.py
├── datasets/
│   └── finance_dataset.json
└── evaluate_finance.py

修改文件:
└── run_llm_compiler.py
```
