# 测试数据收集方案设计

## 目标

收集LLMCompiler的测试数据，重点验证：
1. **DAG生成准确性** - 任务依赖关系是否合理，是否存在循环依赖或缺失依赖
2. **参数传递正确性** - `$id`引用是否正确解析，参数是否正确传递给工具

## 方案概述

采用**增强现有输出 + 独立验证脚本**方案，不修改核心代码，通过后处理分析收集的数据。

## 数据收集

### 运行命令

```bash
export HTTP_PROXY=http://127.0.0.1:10808
export HTTPS_PROXY=http://127.0.0.1:10808

uv run python run_llm_compiler.py --benchmark_name hotpotqa --store results/test_hotpotqa.json --stream --do_benchmark --N 30
uv run python run_llm_compiler.py --benchmark_name movie --store results/test_movie.json --stream --do_benchmark --N 30
uv run python run_llm_compiler.py --benchmark_name parallelqa --store results/test_parallelqa.json --stream --do_benchmark --N 30
```

### 收集的数据

每个样本包含：
- `question`: 原始问题
- `label`: 正确答案
- `answer`: 模型输出
- `time`: 执行时间
- `plan`: DAG结构信息（任务列表、依赖关系、工具调用）
- `stats`: 运行时统计

## 验证脚本结构

### 文件：`validate_test_data.py`

```
validate_test_data.py
├── DAG分析器
│   ├── 解析任务依赖关系
│   ├── 检测循环依赖
│   ├── 计算并行度（最大可并行任务数）
│   └── 识别关键路径
│
├── 参数传递验证器
│   ├── 提取所有 $id 和 ${id} 引用
│   ├── 验证引用的任务是否存在
│   ├── 检查参数类型是否匹配工具签名
│   └── 标记潜在的引用链问题
│
├── 报告生成器
│   ├── 汇总统计（成功率、平均并行度、依赖深度）
│   ├── 问题列表（按严重程度分类）
│   └── 每个样本的详细分析
│
└── 输出格式
    ├── JSON详细报告：results/validation_report.json
    └── Markdown可读报告：results/validation_report.md
```

### 验证规则

**DAG验证：**
1. 无循环依赖（通过拓扑排序检测）
2. 所有依赖引用的任务存在
3. 依赖编号小于当前任务编号（避免前向引用）

**参数验证：**
1. `$id`和`${id}`语法正确
2. 引用的任务ID在有效范围内
3. 参数数量与工具签名匹配

## 输出格式

### JSON报告：`validation_report.json`

```json
{
  "summary": {
    "total_samples": 90,
    "issues_found": 5,
    "by_benchmark": {
      "hotpotqa": {"samples": 30, "issues": 2},
      "movie": {"samples": 30, "issues": 1},
      "parallelqa": {"samples": 30, "issues": 2}
    },
    "avg_parallelism": 2.3,
    "avg_dependency_depth": 3.1
  },
  "issues": [
    {
      "sample_id": "15",
      "benchmark": "hotpotqa",
      "severity": "error",
      "type": "invalid_reference",
      "message": "任务3引用了不存在的$5",
      "location": {"task_idx": 3}
    }
  ],
  "samples": [...]
}
```

### Markdown报告：`validation_report.md`

```markdown
# DAG验证报告

## 概览
- 总样本数: 90
- 问题数: 5
- 平均并行度: 2.3
- 平均依赖深度: 3.1

## 问题列表

### 严重问题 (2)
| 样本ID | 基准测试 | 问题类型 | 描述 |
|--------|----------|----------|------|
| 15 | hotpotqa | 参数引用错误 | 任务3引用了不存在的$5 |

### 警告 (3)
| 样本ID | 基准测试 | 问题类型 | 描述 |
|--------|----------|----------|------|
| 7 | movie | 低并行度 | 所有任务串行执行 |

## 样本详情

### hotpotqa - 样本 1
**问题:** What is the population of...
**DAG结构:**
```
[1] search("city") → 无依赖
[2] search("country") → 无依赖
[3] process($1, $2) → 依赖[1,2]
```
**状态:** 正常
```

## 项目结构

```
LLMCompiler/
├── results/
│   ├── test_hotpotqa.json
│   ├── test_movie.json
│   ├── test_parallelqa.json
│   ├── validation_report.json
│   └── validation_report.md
│
├── validate_test_data.py
│
└── scripts/
    └── collect_test_data.sh
```

## 使用方式

```bash
# 1. 收集数据
bash scripts/collect_test_data.sh

# 2. 验证并生成报告
uv run python validate_test_data.py

# 3. 查看报告
cat results/validation_report.md
```

## 技术细节

### DAG解析

从`plan`字段提取任务信息：
- 任务索引
- 工具名称
- 参数列表
- 依赖列表

### 循环依赖检测

使用Kahn算法进行拓扑排序，若无法完成排序则存在循环。

### 参数引用提取

正则表达式匹配：
- `\$[0-9]+` - 简单引用
- `\$\{[0-9]+\}` - 大括号引用
