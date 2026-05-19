# Finance Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `finance` benchmark to test LLMCompiler's dependency generation and parameter generation precision using complex financial scenarios with DAG-level evaluation.

**Architecture:** Create a new benchmark configuration following existing patterns (configs/finance/), reuse parallelqa tools (search + math), add DAG comparison logic in evaluate_finance.py, and modify run_llm_compiler.py to support the new benchmark.

**Tech Stack:** Python 3.10, networkx for DAG isomorphism, difflib for string similarity

---

## File Structure

```
configs/finance/
├── configs.py           # Import prompts, define CONFIGS dict
├── tools.py             # Re-export parallelqa tools
├── gpt_prompts.py       # PLANNER_PROMPT and OUTPUT_PROMPT for GPT
└── llama_prompts.py     # Same prompts for LLaMA-style models

datasets/
└── finance_dataset.json # 20-30 test samples with expected_dag

evaluate_finance.py      # DAG-level evaluation script
run_llm_compiler.py      # Add finance support (modify)
```

---

### Task 1: Create configs/finance/gpt_prompts.py

**Files:**
- Create: `configs/finance/gpt_prompts.py`

- [ ] **Step 1: Create the gpt_prompts.py file with PLANNER_PROMPT and OUTPUT_PROMPT**

```python
from src.llm_compiler.constants import END_OF_PLAN, JOINNER_FINISH

PLANNER_PROMPT = (
    "Question: 比较苹果公司和微软公司2023年的净利润率，哪家更高？\n"
    '1. search("Apple Inc.")\n'
    '2. search("Microsoft")\n'
    '3. math("Apple net profit margin in 2023 in percentage", ["$1"])\n'
    '4. math("Microsoft net profit margin in 2023 in percentage", ["$2"])\n'
    "Thought: Comparing the profit margins, Microsoft has higher margin.\n"
    f"5. join(){END_OF_PLAN}\n"
    "###\n"
    "\n"
    "Question: 如果亚马逊2022年的营收增长10%，同时谷歌2022年的营收下降5%，两者营收相差多少亿美元？\n"
    '1. search("Amazon revenue 2022")\n'
    '2. search("Alphabet Google revenue 2022")\n'
    '3. math("Amazon revenue in billions if increased by 10%", ["$1"])\n'
    '4. math("Google revenue in billions if decreased by 5%", ["$2"])\n'
    '5. math("absolute difference between $3 and $4 in billions", ["$3", "$4"])\n'
    "Thought: I can answer the question now.\n"
    f"6. join(){END_OF_PLAN}\n"
    "###\n"
    "\n"
    "Question: 计算特斯拉、蔚来、小鹏三家电动汽车公司2023年的市值总和，并求平均值。\n"
    '1. search("Tesla market cap 2023")\n'
    '2. search("NIO market cap 2023")\n'
    '3. search("XPeng market cap 2023")\n'
    '4. math("Tesla market cap in billions", ["$1"])\n'
    '5. math("NIO market cap in billions", ["$2"])\n'
    '6. math("XPeng market cap in billions", ["$3"])\n'
    '7. math("sum of $4, $5, $6 divided by 3 in billions", ["$4", "$5", "$6"])\n'
    "Thought: I can answer the question now.\n"
    f"8. join(){END_OF_PLAN}\n"
    "###\n"
    "\n"
    "Question: 比较中美两国2022年的GDP，并计算其差额占美国GDP的百分比。\n"
    '1. search("China GDP 2022")\n'
    '2. search("United States GDP 2022")\n'
    '3. math("China GDP in trillions", ["$1"])\n'
    '4. math("United States GDP in trillions", ["$2"])\n'
    '5. math("(($4 - $3) / $4) * 100 in percentage", ["$3", "$4"])\n'
    "Thought: I can answer the question now.\n"
    f"6. join(){END_OF_PLAN}\n"
    "###\n"
    "\n"
    "Question: 如果英伟达2023年股价上涨50%，AMD上涨30%，谁的市值更高？\n"
    '1. search("Nvidia market cap 2023")\n'
    '2. search("AMD market cap 2023")\n'
    '3. math("Nvidia market cap in billions if increased by 50%", ["$1"])\n'
    '4. math("AMD market cap in billions if increased by 30%", ["$2"])\n'
    "Thought: I can compare the numbers now.\n"
    f"5. join(){END_OF_PLAN}\n"
    "###\n"
)

OUTPUT_PROMPT = (
    "Solve a financial question answering task with interleaving Observation, Thought, and Action steps. "
    "Answer should always be a single item and MUST not be multiple choices.\n"
    "Thought step can reason about the observations in a few words. You MUST keep it short.\n"
    "Action can be only one type:"
    f" (1) {JOINNER_FINISH}(answer): returns the answer and finishes the task. "
    "    - Final answer MUST NOT contain any description, and must be short (e.g. company names, numbers, Yes/No, etc.)\n"
    "    - When comparing companies or metrics, return the name of the company or the specific value asked.\n"
    "    - If you are asked about a value (e.g. ratio, difference, average, ...), it has to be a number, not a description.\n"
    "\n"
    "Here are some examples:\n"
    "\n"
    "Question: 比较苹果公司和微软公司2023年的净利润率，哪家更高？\n"
    "search(Apple Inc.)\n"
    "Observation: Apple Inc. is a technology company... net income $94.68 billion in 2022 on revenue of $394.33 billion...\n"
    "search(Microsoft)\n"
    "Observation: Microsoft Corporation is a technology company... net income $72.74 billion in 2022 on revenue of $198.27 billion...\n"
    "math(Apple net profit margin in 2023 in percentage)\n"
    "Observation: 25.3\n"
    "math(Microsoft net profit margin in 2023 in percentage)\n"
    "Observation: 36.4\n"
    "Thought: Microsoft has higher profit margin at 36.4% vs Apple's 25.3%.\n"
    f"Action: {JOINNER_FINISH}(Microsoft)\n"
    "###\n"
    "\n"
    "Question: 如果亚马逊2022年的营收增长10%，同时谷歌2022年的营收下降5%，两者营收相差多少亿美元？\n"
    "search(Amazon revenue 2022)\n"
    "Observation: Amazon reported revenue of $514 billion in 2022...\n"
    "search(Alphabet Google revenue 2022)\n"
    "Observation: Alphabet reported revenue of $282.8 billion in 2022...\n"
    "math(Amazon revenue in billions if increased by 10%)\n"
    "Observation: 565.4\n"
    "math(Google revenue in billions if decreased by 5%)\n"
    "Observation: 268.66\n"
    "math(absolute difference between 565.4 and 268.66 in billions)\n"
    "Observation: 296.74\n"
    "Thought: The difference is 296.74 billion.\n"
    f"Action: {JOINNER_FINISH}(296.74)\n"
    "###\n"
)
```

- [ ] **Step 2: Commit**

```bash
git add configs/finance/gpt_prompts.py
git commit -m "feat(finance): add gpt_prompts.py with PLANNER_PROMPT and OUTPUT_PROMPT"
```

---

### Task 2: Create configs/finance/llama_prompts.py

**Files:**
- Create: `configs/finance/llama_prompts.py`

- [ ] **Step 1: Create the llama_prompts.py file (same content as gpt_prompts.py for now)**

```python
from configs.finance.gpt_prompts import PLANNER_PROMPT, OUTPUT_PROMPT

# For LLaMA-style models, we use the same prompts as GPT.
# Can be customized later if needed.
```

- [ ] **Step 2: Commit**

```bash
git add configs/finance/llama_prompts.py
git commit -m "feat(finance): add llama_prompts.py reusing gpt_prompts"
```

---

### Task 3: Create configs/finance/tools.py

**Files:**
- Create: `configs/finance/tools.py`

- [ ] **Step 1: Create the tools.py file reusing parallelqa tools**

```python
"""Finance benchmark tools.

Reuses parallelqa tools (search + math) for financial data queries.
"""

from configs.parallelqa.tools import tools, generate_tools

__all__ = ["tools", "generate_tools"]
```

- [ ] **Step 2: Commit**

```bash
git add configs/finance/tools.py
git commit -m "feat(finance): add tools.py reusing parallelqa tools"
```

---

### Task 4: Create configs/finance/configs.py

**Files:**
- Create: `configs/finance/configs.py`

- [ ] **Step 1: Create the configs.py file**

```python
from configs.finance.gpt_prompts import OUTPUT_PROMPT as GPT_OUTPUT_PROMPT
from configs.finance.gpt_prompts import PLANNER_PROMPT as GPT_PLANNER_PROMPT
from configs.finance.llama_prompts import OUTPUT_PROMPT as LLAMA_OUTPUT_PROMPT
from configs.finance.llama_prompts import PLANNER_PROMPT as LLAMA_PLANNER_PROMPT

CONFIGS = {
    "default_model": "gpt-3.5-turbo-1106",
    "prompts": {
        "gpt": {
            "planner_prompt": GPT_PLANNER_PROMPT,
            "output_prompt": GPT_OUTPUT_PROMPT,
        },
        "llama": {
            "planner_prompt": LLAMA_PLANNER_PROMPT,
            "output_prompt": LLAMA_OUTPUT_PROMPT,
        },
    },
    "max_replans": 1,
}
```

- [ ] **Step 2: Commit**

```bash
git add configs/finance/configs.py
git commit -m "feat(finance): add configs.py with CONFIGS dict"
```

---

### Task 5: Create datasets/finance_dataset.json

**Files:**
- Create: `datasets/finance_dataset.json`

- [ ] **Step 1: Create the finance_dataset.json with 25 test samples**

```json
[
    {
        "id": "1",
        "question": "比较苹果公司和微软公司2023年的净利润率，哪家更高？",
        "answer": "微软",
        "complexity": "deep",
        "expected_dag": {
            "tasks": [
                {"idx": 1, "tool": "search", "args": ["Apple Inc."]},
                {"idx": 2, "tool": "search", "args": ["Microsoft"]},
                {"idx": 3, "tool": "math", "args": ["Apple net profit margin in 2023 in percentage", ["$1"]]},
                {"idx": 4, "tool": "math", "args": ["Microsoft net profit margin in 2023 in percentage", ["$2"]]},
                {"idx": 5, "tool": "join", "args": []}
            ],
            "dependencies": {"3": [1], "4": [2], "5": [3, 4]}
        }
    },
    {
        "id": "2",
        "question": "如果亚马逊2022年的营收增长10%，同时谷歌2022年的营收下降5%，两者营收相差多少亿美元？",
        "answer": "296.74",
        "complexity": "deep",
        "expected_dag": {
            "tasks": [
                {"idx": 1, "tool": "search", "args": ["Amazon revenue 2022"]},
                {"idx": 2, "tool": "search", "args": ["Alphabet Google revenue 2022"]},
                {"idx": 3, "tool": "math", "args": ["Amazon revenue in billions if increased by 10%", ["$1"]]},
                {"idx": 4, "tool": "math", "args": ["Google revenue in billions if decreased by 5%", ["$2"]]},
                {"idx": 5, "tool": "math", "args": ["absolute difference between $3 and $4 in billions", ["$3", "$4"]]},
                {"idx": 6, "tool": "join", "args": []}
            ],
            "dependencies": {"3": [1], "4": [2], "5": [3, 4], "6": [5]}
        }
    },
    {
        "id": "3",
        "question": "计算特斯拉、蔚来、小鹏三家电动汽车公司2023年的市值总和，并求平均值。",
        "answer": "650.67",
        "complexity": "deep",
        "expected_dag": {
            "tasks": [
                {"idx": 1, "tool": "search", "args": ["Tesla market cap 2023"]},
                {"idx": 2, "tool": "search", "args": ["NIO market cap 2023"]},
                {"idx": 3, "tool": "search", "args": ["XPeng market cap 2023"]},
                {"idx": 4, "tool": "math", "args": ["Tesla market cap in billions", ["$1"]]},
                {"idx": 5, "tool": "math", "args": ["NIO market cap in billions", ["$2"]]},
                {"idx": 6, "tool": "math", "args": ["XPeng market cap in billions", ["$3"]]},
                {"idx": 7, "tool": "math", "args": ["sum of $4, $5, $6 divided by 3 in billions", ["$4", "$5", "$6"]]},
                {"idx": 8, "tool": "join", "args": []}
            ],
            "dependencies": {"4": [1], "5": [2], "6": [3], "7": [4, 5, 6], "8": [7]}
        }
    },
    {
        "id": "4",
        "question": "比较中美两国2022年的GDP，并计算其差额占美国GDP的百分比。",
        "answer": "35.5",
        "complexity": "deep",
        "expected_dag": {
            "tasks": [
                {"idx": 1, "tool": "search", "args": ["China GDP 2022"]},
                {"idx": 2, "tool": "search", "args": ["United States GDP 2022"]},
                {"idx": 3, "tool": "math", "args": ["China GDP in trillions", ["$1"]]},
                {"idx": 4, "tool": "math", "args": ["United States GDP in trillions", ["$2"]]},
                {"idx": 5, "tool": "math", "args": ["(($4 - $3) / $4) * 100 in percentage", ["$3", "$4"]]},
                {"idx": 6, "tool": "join", "args": []}
            ],
            "dependencies": {"3": [1], "4": [2], "5": [3, 4], "6": [5]}
        }
    },
    {
        "id": "5",
        "question": "如果英伟达2023年股价上涨50%，AMD上涨30%，谁的市值更高？",
        "answer": "英伟达",
        "complexity": "medium",
        "expected_dag": {
            "tasks": [
                {"idx": 1, "tool": "search", "args": ["Nvidia market cap 2023"]},
                {"idx": 2, "tool": "search", "args": ["AMD market cap 2023"]},
                {"idx": 3, "tool": "math", "args": ["Nvidia market cap in billions if increased by 50%", ["$1"]]},
                {"idx": 4, "tool": "math", "args": ["AMD market cap in billions if increased by 30%", ["$2"]]},
                {"idx": 5, "tool": "join", "args": []}
            ],
            "dependencies": {"3": [1], "4": [2], "5": [3, 4]}
        }
    },
    {
        "id": "6",
        "question": "比较苹果和三星2023年的研发支出，哪个更高？",
        "answer": "苹果",
        "complexity": "shallow",
        "expected_dag": {
            "tasks": [
                {"idx": 1, "tool": "search", "args": ["Apple R&D expenditure 2023"]},
                {"idx": 2, "tool": "search", "args": ["Samsung R&D expenditure 2023"]},
                {"idx": 3, "tool": "join", "args": []}
            ],
            "dependencies": {"3": [1, 2]}
        }
    },
    {
        "id": "7",
        "question": "计算腾讯、阿里巴巴、京东三家公司2023年营收的总和。",
        "answer": "1050",
        "complexity": "deep",
        "expected_dag": {
            "tasks": [
                {"idx": 1, "tool": "search", "args": ["Tencent revenue 2023"]},
                {"idx": 2, "tool": "search", "args": ["Alibaba revenue 2023"]},
                {"idx": 3, "tool": "search", "args": ["JD.com revenue 2023"]},
                {"idx": 4, "tool": "math", "args": ["Tencent revenue in billions", ["$1"]]},
                {"idx": 5, "tool": "math", "args": ["Alibaba revenue in billions", ["$2"]]},
                {"idx": 6, "tool": "math", "args": ["JD.com revenue in billions", ["$3"]]},
                {"idx": 7, "tool": "math", "args": ["sum of $4, $5, $6 in billions", ["$4", "$5", "$6"]]},
                {"idx": 8, "tool": "join", "args": []}
            ],
            "dependencies": {"4": [1], "5": [2], "6": [3], "7": [4, 5, 6], "8": [7]}
        }
    },
    {
        "id": "8",
        "question": "如果Meta的2023年净利润增长20%，其净利润将达到多少亿美元？",
        "answer": "46.8",
        "complexity": "shallow",
        "expected_dag": {
            "tasks": [
                {"idx": 1, "tool": "search", "args": ["Meta net income 2023"]},
                {"idx": 2, "tool": "math", "args": ["Meta net income in billions if increased by 20%", ["$1"]]},
                {"idx": 3, "tool": "join", "args": []}
            ],
            "dependencies": {"2": [1], "3": [2]}
        }
    },
    {
        "id": "9",
        "question": "比较美国和日本2022年的通胀率，哪个更高？",
        "answer": "美国",
        "complexity": "shallow",
        "expected_dag": {
            "tasks": [
                {"idx": 1, "tool": "search", "args": ["United States inflation rate 2022"]},
                {"idx": 2, "tool": "search", "args": ["Japan inflation rate 2022"]},
                {"idx": 3, "tool": "join", "args": []}
            ],
            "dependencies": {"3": [1, 2]}
        }
    },
    {
        "id": "10",
        "question": "计算伯克希尔哈撒韦和黑石集团2023年管理资产规模的比率。",
        "answer": "3.2",
        "complexity": "medium",
        "expected_dag": {
            "tasks": [
                {"idx": 1, "tool": "search", "args": ["Berkshire Hathaway assets 2023"]},
                {"idx": 2, "tool": "search", "args": ["BlackRock assets under management 2023"]},
                {"idx": 3, "tool": "math", "args": ["Berkshire Hathaway assets in billions", ["$1"]]},
                {"idx": 4, "tool": "math", "args": ["BlackRock AUM in billions", ["$2"]]},
                {"idx": 5, "tool": "math", "args": ["ratio of $3 to $4", ["$3", "$4"]]},
                {"idx": 6, "tool": "join", "args": []}
            ],
            "dependencies": {"3": [1], "4": [2], "5": [3, 4], "6": [5]}
        }
    },
    {
        "id": "11",
        "question": "如果比亚迪2023年销量增长50%，其销量将达到多少万辆？",
        "answer": "210",
        "complexity": "shallow",
        "expected_dag": {
            "tasks": [
                {"idx": 1, "tool": "search", "args": ["BYD sales 2023"]},
                {"idx": 2, "tool": "math", "args": ["BYD sales in ten thousands if increased by 50%", ["$1"]]},
                {"idx": 3, "tool": "join", "args": []}
            ],
            "dependencies": {"2": [1], "3": [2]}
        }
    },
    {
        "id": "12",
        "question": "比较美联储和欧洲央行2023年的基准利率，哪个更高？",
        "answer": "美联储",
        "complexity": "shallow",
        "expected_dag": {
            "tasks": [
                {"idx": 1, "tool": "search", "args": ["Federal Reserve interest rate 2023"]},
                {"idx": 2, "tool": "search", "args": ["European Central Bank interest rate 2023"]},
                {"idx": 3, "tool": "join", "args": []}
            ],
            "dependencies": {"3": [1, 2]}
        }
    },
    {
        "id": "13",
        "question": "计算高盛、摩根士丹利、摩根大通三家投行2023年净利润的平均值。",
        "answer": "120",
        "complexity": "deep",
        "expected_dag": {
            "tasks": [
                {"idx": 1, "tool": "search", "args": ["Goldman Sachs net income 2023"]},
                {"idx": 2, "tool": "search", "args": ["Morgan Stanley net income 2023"]},
                {"idx": 3, "tool": "search", "args": ["JPMorgan net income 2023"]},
                {"idx": 4, "tool": "math", "args": ["Goldman Sachs net income in billions", ["$1"]]},
                {"idx": 5, "tool": "math", "args": ["Morgan Stanley net income in billions", ["$2"]]},
                {"idx": 6, "tool": "math", "args": ["JPMorgan net income in billions", ["$3"]]},
                {"idx": 7, "tool": "math", "args": ["average of $4, $5, $6 in billions", ["$4", "$5", "$6"]]},
                {"idx": 8, "tool": "join", "args": []}
            ],
            "dependencies": {"4": [1], "5": [2], "6": [3], "7": [4, 5, 6], "8": [7]}
        }
    },
    {
        "id": "14",
        "question": "如果丰田2023年营收下降10%，而大众增长15%，两者的营收差距是多少亿美元？",
        "answer": "85",
        "complexity": "medium",
        "expected_dag": {
            "tasks": [
                {"idx": 1, "tool": "search", "args": ["Toyota revenue 2023"]},
                {"idx": 2, "tool": "search", "args": ["Volkswagen revenue 2023"]},
                {"idx": 3, "tool": "math", "args": ["Toyota revenue in billions if decreased by 10%", ["$1"]]},
                {"idx": 4, "tool": "math", "args": ["Volkswagen revenue in billions if increased by 15%", ["$2"]]},
                {"idx": 5, "tool": "math", "args": ["absolute difference between $3 and $4 in billions", ["$3", "$4"]]},
                {"idx": 6, "tool": "join", "args": []}
            ],
            "dependencies": {"3": [1], "4": [2], "5": [3, 4], "6": [5]}
        }
    },
    {
        "id": "15",
        "question": "比较沙特阿美和埃克森美孚2023年的营收，哪家更高？",
        "answer": "沙特阿美",
        "complexity": "shallow",
        "expected_dag": {
            "tasks": [
                {"idx": 1, "tool": "search", "args": ["Saudi Aramco revenue 2023"]},
                {"idx": 2, "tool": "search", "args": ["ExxonMobil revenue 2023"]},
                {"idx": 3, "tool": "join", "args": []}
            ],
            "dependencies": {"3": [1, 2]}
        }
    },
    {
        "id": "16",
        "question": "计算微软、谷歌、亚马逊三家公司2023年云计算业务收入的总和。",
        "answer": "150",
        "complexity": "deep",
        "expected_dag": {
            "tasks": [
                {"idx": 1, "tool": "search", "args": ["Microsoft Azure revenue 2023"]},
                {"idx": 2, "tool": "search", "args": ["Google Cloud revenue 2023"]},
                {"idx": 3, "tool": "search", "args": ["Amazon AWS revenue 2023"]},
                {"idx": 4, "tool": "math", "args": ["Microsoft Azure revenue in billions", ["$1"]]},
                {"idx": 5, "tool": "math", "args": ["Google Cloud revenue in billions", ["$2"]]},
                {"idx": 6, "tool": "math", "args": ["Amazon AWS revenue in billions", ["$3"]]},
                {"idx": 7, "tool": "math", "args": ["sum of $4, $5, $6 in billions", ["$4", "$5", "$6"]]},
                {"idx": 8, "tool": "join", "args": []}
            ],
            "dependencies": {"4": [1], "5": [2], "6": [3], "7": [4, 5, 6], "8": [7]}
        }
    },
    {
        "id": "17",
        "question": "如果香港恒生指数2023年上涨15%，而日经225上涨25%，哪个指数涨幅更大？",
        "answer": "日经225",
        "complexity": "shallow",
        "expected_dag": {
            "tasks": [
                {"idx": 1, "tool": "search", "args": ["Hang Seng Index 2023 performance"]},
                {"idx": 2, "tool": "search", "args": ["Nikkei 225 2023 performance"]},
                {"idx": 3, "tool": "join", "args": []}
            ],
            "dependencies": {"3": [1, 2]}
        }
    },
    {
        "id": "18",
        "question": "比较台积电和三星电子2023年的资本支出，哪个更高？",
        "answer": "台积电",
        "complexity": "shallow",
        "expected_dag": {
            "tasks": [
                {"idx": 1, "tool": "search", "args": ["TSMC capital expenditure 2023"]},
                {"idx": 2, "tool": "search", "args": ["Samsung Electronics capex 2023"]},
                {"idx": 3, "tool": "join", "args": []}
            ],
            "dependencies": {"3": [1, 2]}
        }
    },
    {
        "id": "19",
        "question": "计算美国、中国、日本、德国四国2022年贸易顺差的总和。",
        "answer": "850",
        "complexity": "deep",
        "expected_dag": {
            "tasks": [
                {"idx": 1, "tool": "search", "args": ["United States trade balance 2022"]},
                {"idx": 2, "tool": "search", "args": ["China trade balance 2022"]},
                {"idx": 3, "tool": "search", "args": ["Japan trade balance 2022"]},
                {"idx": 4, "tool": "search", "args": ["Germany trade balance 2022"]},
                {"idx": 5, "tool": "math", "args": ["US trade balance in billions", ["$1"]]},
                {"idx": 6, "tool": "math", "args": ["China trade balance in billions", ["$2"]]},
                {"idx": 7, "tool": "math", "args": ["Japan trade balance in billions", ["$3"]]},
                {"idx": 8, "tool": "math", "args": ["Germany trade balance in billions", ["$4"]]},
                {"idx": 9, "tool": "math", "args": ["sum of $5, $6, $7, $8 in billions", ["$5", "$6", "$7", "$8"]]},
                {"idx": 10, "tool": "join", "args": []}
            ],
            "dependencies": {"5": [1], "6": [2], "7": [3], "8": [4], "9": [5, 6, 7, 8], "10": [9]}
        }
    },
    {
        "id": "20",
        "question": "如果雪佛龙2023年利润下降20%，其利润将为多少亿美元？",
        "answer": "28",
        "complexity": "shallow",
        "expected_dag": {
            "tasks": [
                {"idx": 1, "tool": "search", "args": ["Chevron profit 2023"]},
                {"idx": 2, "tool": "math", "args": ["Chevron profit in billions if decreased by 20%", ["$1"]]},
                {"idx": 3, "tool": "join", "args": []}
            ],
            "dependencies": {"2": [1], "3": [2]}
        }
    },
    {
        "id": "21",
        "question": "比较维萨和万事达卡2023年的交易量，哪个更高？",
        "answer": "维萨",
        "complexity": "shallow",
        "expected_dag": {
            "tasks": [
                {"idx": 1, "tool": "search", "args": ["Visa transaction volume 2023"]},
                {"idx": 2, "tool": "search", "args": ["Mastercard transaction volume 2023"]},
                {"idx": 3, "tool": "join", "args": []}
            ],
            "dependencies": {"3": [1, 2]}
        }
    },
    {
        "id": "22",
        "question": "计算英特尔、AMD、英伟达三家公司2023年研发支出占总收入比率的平均值。",
        "answer": "22",
        "complexity": "deep",
        "expected_dag": {
            "tasks": [
                {"idx": 1, "tool": "search", "args": ["Intel R&D and revenue 2023"]},
                {"idx": 2, "tool": "search", "args": ["AMD R&D and revenue 2023"]},
                {"idx": 3, "tool": "search", "args": ["Nvidia R&D and revenue 2023"]},
                {"idx": 4, "tool": "math", "args": ["Intel R&D as percentage of revenue", ["$1"]]},
                {"idx": 5, "tool": "math", "args": ["AMD R&D as percentage of revenue", ["$2"]]},
                {"idx": 6, "tool": "math", "args": ["Nvidia R&D as percentage of revenue", ["$3"]]},
                {"idx": 7, "tool": "math", "args": ["average of $4, $5, $6 in percentage", ["$4", "$5", "$6"]]},
                {"idx": 8, "tool": "join", "args": []}
            ],
            "dependencies": {"4": [1], "5": [2], "6": [3], "7": [4, 5, 6], "8": [7]}
        }
    },
    {
        "id": "23",
        "question": "比较波音和空客2023年的飞机交付量，哪个更多？",
        "answer": "空客",
        "complexity": "shallow",
        "expected_dag": {
            "tasks": [
                {"idx": 1, "tool": "search", "args": ["Boeing deliveries 2023"]},
                {"idx": 2, "tool": "search", "args": ["Airbus deliveries 2023"]},
                {"idx": 3, "tool": "join", "args": []}
            ],
            "dependencies": {"3": [1, 2]}
        }
    },
    {
        "id": "24",
        "question": "如果辉瑞2023年营收下降40%，而默克增长10%，两者的营收比率是多少？",
        "answer": "0.85",
        "complexity": "medium",
        "expected_dag": {
            "tasks": [
                {"idx": 1, "tool": "search", "args": ["Pfizer revenue 2023"]},
                {"idx": 2, "tool": "search", "args": ["Merck revenue 2023"]},
                {"idx": 3, "tool": "math", "args": ["Pfizer revenue in billions if decreased by 40%", ["$1"]]},
                {"idx": 4, "tool": "math", "args": ["Merck revenue in billions if increased by 10%", ["$2"]]},
                {"idx": 5, "tool": "math", "args": ["ratio of $3 to $4", ["$3", "$4"]]},
                {"idx": 6, "tool": "join", "args": []}
            ],
            "dependencies": {"3": [1], "4": [2], "5": [3, 4], "6": [5]}
        }
    },
    {
        "id": "25",
        "question": "计算比特币、以太坊、瑞波币三种加密货币2023年底市值的总和。",
        "answer": "1050",
        "complexity": "deep",
        "expected_dag": {
            "tasks": [
                {"idx": 1, "tool": "search", "args": ["Bitcoin market cap 2023"]},
                {"idx": 2, "tool": "search", "args": ["Ethereum market cap 2023"]},
                {"idx": 3, "tool": "search", "args": ["XRP market cap 2023"]},
                {"idx": 4, "tool": "math", "args": ["Bitcoin market cap in billions", ["$1"]]},
                {"idx": 5, "tool": "math", "args": ["Ethereum market cap in billions", ["$2"]]},
                {"idx": 6, "tool": "math", "args": ["XRP market cap in billions", ["$3"]]},
                {"idx": 7, "tool": "math", "args": ["sum of $4, $5, $6 in billions", ["$4", "$5", "$6"]]},
                {"idx": 8, "tool": "join", "args": []}
            ],
            "dependencies": {"4": [1], "5": [2], "6": [3], "7": [4, 5, 6], "8": [7]}
        }
    }
]
```

- [ ] **Step 2: Commit**

```bash
git add datasets/finance_dataset.json
git commit -m "feat(finance): add finance_dataset.json with 25 test samples"
```

---

### Task 6: Modify run_llm_compiler.py to support finance benchmark

**Files:**
- Modify: `run_llm_compiler.py`

- [ ] **Step 1: Add finance imports**

Add after line 27 (after parallelqa_react_generate_tools import):

```python
from configs.finance.configs import CONFIGS as FINANCE_CONFIGS
from configs.finance.tools import generate_tools as finance_generate_tools
```

- [ ] **Step 2: Update benchmark_name choices**

Change line 55-56 from:
```python
    choices=["movie", "hotpotqa", "parallelqa"],
```
to:
```python
    choices=["movie", "hotpotqa", "parallelqa", "finance"],
```

- [ ] **Step 3: Add finance to get_dataset function**

Add after line 87 (after parallelqa dataset):

```python
    elif args.benchmark_name == "finance":
        dataset_name = "datasets/finance_dataset.json"
```

- [ ] **Step 4: Add finance to get_tools function**

Add after line 106 (after parallelqa tools):

```python
    elif args.benchmark_name == "finance":
        if args.react:
            raise ValueError("Finance benchmark does not support ReAct mode")
        else:
            tools = finance_generate_tools(args, model_name)
```

- [ ] **Step 5: Add finance to get_configs function**

Add after line 127 (after parallelqa configs):

```python
    elif args.benchmark_name == "finance":
        if args.react:
            raise ValueError("Finance benchmark does not support ReAct mode")
        else:
            configs = FINANCE_CONFIGS
```

- [ ] **Step 6: Commit**

```bash
git add run_llm_compiler.py
git commit -m "feat(finance): integrate finance benchmark into run_llm_compiler.py"
```

---

### Task 7: Create evaluate_finance.py

**Files:**
- Create: `evaluate_finance.py`

- [ ] **Step 1: Create the evaluate_finance.py file with DAG comparison logic**

```python
#!/usr/bin/env python3
"""Evaluate finance benchmark results with DAG-level metrics.

This script compares generated DAGs against expected DAGs to measure
dependency generation and parameter generation precision.
"""

import argparse
import json
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx


def string_similarity(s1: str, s2: str) -> float:
    """Calculate string similarity ratio (0-1)."""
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def normalize_args(args: Any) -> Tuple:
    """Normalize args for comparison."""
    if isinstance(args, str):
        return (args,)
    elif isinstance(args, list):
        return tuple(args)
    elif isinstance(args, tuple):
        return args
    return (str(args),)


def extract_task_signature(task: Dict) -> Tuple:
    """Extract a comparable signature from a task."""
    tool = task.get("name", task.get("tool", ""))
    args = task.get("args", [])
    return (tool, normalize_args(args))


def match_tasks(
    generated_tasks: List[Dict],
    expected_tasks: List[Dict],
    similarity_threshold: float = 0.8,
) -> Dict[int, Optional[int]]:
    """Match generated tasks to expected tasks using Hungarian-like matching.

    Returns:
        Dict mapping generated task idx -> expected task idx (or None if no match)
    """
    # Build similarity matrix
    gen_by_idx = {t["idx"]: t for t in generated_tasks}
    exp_by_idx = {t["idx"]: t for t in expected_tasks}

    gen_indices = list(gen_by_idx.keys())
    exp_indices = list(exp_by_idx.keys())

    # Simple greedy matching (can be improved with Hungarian algorithm)
    matches = {}
    used_expected = set()

    for gen_idx in gen_indices:
        gen_task = gen_by_idx[gen_idx]
        gen_tool = gen_task.get("name", gen_task.get("tool", ""))
        gen_args = gen_task.get("args", [])

        best_match = None
        best_score = similarity_threshold

        for exp_idx in exp_indices:
            if exp_idx in used_expected:
                continue

            exp_task = exp_by_idx[exp_idx]
            exp_tool = exp_task.get("tool", "")
            exp_args = exp_task.get("args", [])

            # Tool must match exactly
            if gen_tool != exp_tool:
                continue

            # Calculate arg similarity
            if gen_tool == "search":
                # For search, compare the query string
                if gen_args and exp_args:
                    score = string_similarity(str(gen_args[0]), str(exp_args[0]))
                    if score > best_score:
                        best_score = score
                        best_match = exp_idx
            elif gen_tool == "math":
                # For math, compare problem string and context references
                if gen_args and exp_args:
                    # Compare problem string
                    gen_problem = str(gen_args[0]) if gen_args else ""
                    exp_problem = str(exp_args[0]) if exp_args else ""
                    problem_score = string_similarity(gen_problem, exp_problem)

                    # Check context references match
                    gen_context = set(gen_args[1]) if len(gen_args) > 1 else set()
                    exp_context = set(exp_args[1]) if len(exp_args) > 1 else set()

                    # Context must match exactly for good match
                    if gen_context == exp_context and problem_score > best_score:
                        best_score = problem_score
                        best_match = exp_idx
                    elif problem_score > best_score + 0.1:  # Allow looser match
                        best_score = problem_score
                        best_match = exp_idx
            elif gen_tool == "join":
                best_match = exp_idx
                best_score = 1.0

        if best_match is not None:
            matches[gen_idx] = best_match
            used_expected.add(best_match)
        else:
            matches[gen_idx] = None

    return matches


def compare_dependencies(
    generated_deps: Dict[str, List[int]],
    expected_deps: Dict[str, List[int]],
    task_matches: Dict[int, Optional[int]],
) -> Tuple[int, int, int, int]:
    """Compare dependency edges.

    Returns:
        (correct_edges, missing_edges, extra_edges, total_expected)
    """
    # Build reverse mapping: expected idx -> generated idx
    reverse_matches = {v: k for k, v in task_matches.items() if v is not None}

    correct = 0
    missing = 0
    extra = 0
    total_expected = 0

    # Count expected edges
    for depender, dependees in expected_deps.items():
        total_expected += len(dependees)

    # Count generated edges vs expected
    for gen_depender, gen_dependees in generated_deps.items():
        gen_depender_int = int(gen_depender) if isinstance(gen_depender, str) else gen_depender

        for gen_dependee in gen_dependees:
            # Check if this edge exists in expected
            matched_depender = task_matches.get(gen_depender_int)
            matched_dependee = task_matches.get(gen_dependee)

            if matched_depender is None or matched_dependee is None:
                extra += 1
                continue

            # Check if expected has this edge
            expected_dependees = expected_deps.get(str(matched_depender), [])
            if matched_dependee in expected_dependees:
                correct += 1
            else:
                extra += 1

    # Count missing edges
    for exp_depender, exp_dependees in expected_deps.items():
        exp_depender_int = int(exp_depender)
        gen_depender = reverse_matches.get(exp_depender_int)

        if gen_depender is None:
            missing += len(exp_dependees)
            continue

        gen_dependees = generated_deps.get(str(gen_depender), [])

        for exp_dependee in exp_dependees:
            gen_dependee = reverse_matches.get(exp_dependee)
            if gen_dependee is None or gen_dependee not in gen_dependees:
                missing += 1

    return correct, missing, extra, total_expected


def check_dag_isomorphism(
    generated_tasks: List[Dict],
    generated_deps: Dict[str, List[int]],
    expected_tasks: List[Dict],
    expected_deps: Dict[str, List[int]],
    task_matches: Dict[int, Optional[int]],
) -> bool:
    """Check if generated DAG is isomorphic to expected DAG."""
    if len(generated_tasks) != len(expected_tasks):
        return False

    # Build NetworkX graphs
    gen_G = nx.DiGraph()
    exp_G = nx.DiGraph()

    # Add nodes with attributes
    for task in generated_tasks:
        gen_G.add_node(task["idx"], tool=task.get("name", task.get("tool", "")))

    for task in expected_tasks:
        exp_G.add_node(task["idx"], tool=task.get("tool", ""))

    # Add edges
    for depender, dependees in generated_deps.items():
        for dependee in dependees:
            gen_G.add_edge(dependee, int(depender))

    for depender, dependees in expected_deps.items():
        for dependee in dependees:
            exp_G.add_edge(dependee, int(depender))

    # Check isomorphism
    try:
        return nx.is_isomorphic(gen_G, exp_G, node_match=lambda n1, n2: n1.get("tool") == n2.get("tool"))
    except Exception:
        return False


def evaluate_arg_references(
    generated_tasks: List[Dict],
    expected_tasks: List[Dict],
    task_matches: Dict[int, Optional[int]],
) -> Tuple[int, int]:
    """Evaluate argument reference accuracy.

    Returns:
        (correct_refs, total_refs_needed)
    """
    correct = 0
    total = 0

    for gen_task in generated_tasks:
        gen_idx = gen_task["idx"]
        matched_exp_idx = task_matches.get(gen_idx)

        if matched_exp_idx is None:
            continue

        # Find expected task
        exp_task = None
        for t in expected_tasks:
            if t["idx"] == matched_exp_idx:
                exp_task = t
                break

        if exp_task is None:
            continue

        gen_args = gen_task.get("args", [])
        exp_args = exp_task.get("args", [])

        # Check if this task needs references
        if gen_task.get("name", gen_task.get("tool", "")) == "math" and len(exp_args) > 1:
            total += 1
            gen_context = set(gen_args[1]) if len(gen_args) > 1 else set()
            exp_context = set(exp_args[1]) if len(exp_args) > 1 else set()

            if gen_context == exp_context:
                correct += 1

    return correct, total


def build_dependency_dict(tasks: List[Dict]) -> Dict[str, List[int]]:
    """Build dependency dict from tasks list.

    Each task has a 'dependencies' list indicating which tasks it depends on.
    Returns dict mapping depender idx -> list of dependee idxs.
    """
    deps = {}
    for task in tasks:
        depender = str(task["idx"])
        dependees = task.get("dependencies", [])
        if dependees:
            deps[depender] = dependees
    return deps


def evaluate_sample(
    generated: Dict,
    expected: Dict,
) -> Dict[str, Any]:
    """Evaluate a single sample."""
    gen_plan = generated.get("plan", {})
    exp_dag = expected.get("expected_dag", {})

    gen_tasks = gen_plan.get("tasks", [])
    gen_deps = build_dependency_dict(gen_tasks)  # Build from task.dependencies
    exp_tasks = exp_dag.get("tasks", [])
    exp_deps = exp_dag.get("dependencies", {})

    # Match tasks
    task_matches = match_tasks(gen_tasks, exp_tasks)

    # Task metrics
    matched_count = sum(1 for m in task_matches.values() if m is not None)
    task_recall = matched_count / len(exp_tasks) if exp_tasks else 0
    task_precision = matched_count / len(gen_tasks) if gen_tasks else 0

    # Dependency metrics
    correct_deps, missing_deps, extra_deps, total_exp_deps = compare_dependencies(
        gen_deps, exp_deps, task_matches
    )
    dep_recall = correct_deps / total_exp_deps if total_exp_deps > 0 else 0
    dep_precision = correct_deps / (correct_deps + extra_deps) if (correct_deps + extra_deps) > 0 else 0

    # DAG isomorphism
    is_isomorphic = check_dag_isomorphism(gen_tasks, gen_deps, exp_tasks, exp_deps, task_matches)

    # Arg reference accuracy
    correct_args, total_args = evaluate_arg_references(gen_tasks, exp_tasks, task_matches)
    arg_ref_acc = correct_args / total_args if total_args > 0 else 1.0

    return {
        "task_recall": task_recall,
        "task_precision": task_precision,
        "dep_recall": dep_recall,
        "dep_precision": dep_precision,
        "dag_isomorphic": is_isomorphic,
        "arg_ref_acc": arg_ref_acc,
        "complexity": expected.get("complexity", "unknown"),
    }


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--file", type=str, required=True, help="Results JSON file")
    argparser.add_argument("--detail", action="store_true", help="Show per-sample details")
    argparser.add_argument("--dataset", type=str, default="datasets/finance_dataset.json", help="Dataset file")
    args = argparser.parse_args()

    # Load results
    with open(args.file, "r") as f:
        results = json.load(f)

    # Load dataset
    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    # Build expected lookup
    expected_by_id = {str(item["id"]): item for item in dataset}

    # Evaluate each sample
    all_metrics = []
    by_complexity = defaultdict(list)

    for sample_id, generated in results.items():
        expected = expected_by_id.get(str(sample_id))
        if expected is None:
            print(f"Warning: No expected data for sample {sample_id}")
            continue

        metrics = evaluate_sample(generated, expected)
        metrics["id"] = sample_id
        all_metrics.append(metrics)
        by_complexity[metrics["complexity"]].append(metrics)

        if args.detail:
            print(f"Sample {sample_id}:")
            print(f"  Task Recall: {metrics['task_recall']:.2%}")
            print(f"  Dep Recall: {metrics['dep_recall']:.2%}")
            print(f"  Arg Ref Acc: {metrics['arg_ref_acc']:.2%}")
            print(f"  DAG Match: {metrics['dag_isomorphic']}")
            print()

    # Aggregate metrics
    total_samples = len(all_metrics)

    avg_task_recall = sum(m["task_recall"] for m in all_metrics) / total_samples
    avg_task_precision = sum(m["task_precision"] for m in all_metrics) / total_samples
    avg_dep_recall = sum(m["dep_recall"] for m in all_metrics) / total_samples
    avg_dep_precision = sum(m["dep_precision"] for m in all_metrics) / total_samples
    avg_dag_match = sum(m["dag_isomorphic"] for m in all_metrics) / total_samples
    avg_arg_ref = sum(m["arg_ref_acc"] for m in all_metrics) / total_samples

    # Print report
    print("=" * 50)
    print("Finance Benchmark Evaluation Report")
    print("=" * 50)
    print(f"Total Samples: {total_samples}")
    print()

    # Complexity breakdown
    print("[Complexity Breakdown]")
    for complexity in ["shallow", "medium", "deep"]:
        metrics_list = by_complexity.get(complexity, [])
        if metrics_list:
            dag_match = sum(m["dag_isomorphic"] for m in metrics_list) / len(metrics_list)
            dep_recall = sum(m["dep_recall"] for m in metrics_list) / len(metrics_list)
            arg_acc = sum(m["arg_ref_acc"] for m in metrics_list) / len(metrics_list)
            print(f"  {complexity} ({len(metrics_list)}): DAG Match {dag_match:.0%}, Dep Recall {dep_recall:.0%}, Arg Acc {arg_acc:.0%}")
    print()

    # Overall metrics
    print("[Overall Metrics]")
    print(f"  Task Recall:    {avg_task_recall:.2f}")
    print(f"  Task Precision: {avg_task_precision:.2f}")
    print(f"  Dep Recall:     {avg_dep_recall:.2f}")
    print(f"  Dep Precision:  {avg_dep_precision:.2f}")
    print(f"  Arg Ref Acc:    {avg_arg_ref:.2f}")
    print(f"  DAG Isomorphism: {avg_dag_match:.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add evaluate_finance.py
git commit -m "feat(finance): add evaluate_finance.py with DAG-level metrics"
```

---

### Task 8: Test the finance benchmark

**Files:**
- None (testing only)

- [ ] **Step 1: Verify configs import correctly**

Run: `uv run python -c "from configs.finance.configs import CONFIGS; print(CONFIGS)"`

Expected: Print CONFIGS dict without errors

- [ ] **Step 2: Verify dataset loads correctly**

Run: `uv run python -c "import json; data = json.load(open('datasets/finance_dataset.json')); print(f'Loaded {len(data)} samples')"`

Expected: `Loaded 25 samples`

- [ ] **Step 3: Run a quick test with finance benchmark (optional, requires API key)**

Run: `uv run python run_llm_compiler.py --benchmark_name finance --store test_finance_results.json --N 1 --stream --model_name deepseek-chat`

Expected: Runs without errors, creates test_finance_results.json

- [ ] **Step 4: Test evaluate_finance.py**

Run: `uv run python evaluate_finance.py --file test_finance_results.json --detail`

Expected: Prints evaluation report

---

### Task 9: Final commit and cleanup

**Files:**
- None (cleanup only)

- [ ] **Step 1: Remove test results file if created**

Run: `rm -f test_finance_results.json`

- [ ] **Step 2: Verify all files are committed**

Run: `git status`

Expected: Working directory clean

- [ ] **Step 3: Show final file structure**

Run: `ls -la configs/finance/ datasets/finance_dataset.json evaluate_finance.py`

Expected: All files present
