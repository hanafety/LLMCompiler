#!/bin/bash
# 数据收集脚本 - 在三个基准测试上运行LLMCompiler

set -e

# 设置代理
export HTTP_PROXY=http://127.0.0.1:10808
export HTTPS_PROXY=http://127.0.0.1:10808

# 样本数量
N_SAMPLES=${1:-30}

# 模型名称（从.env读取或使用默认值）
MODEL_NAME=${MODEL_NAME:-deepseek-v4-flash}

echo "=== 收集测试数据 (样本数: $N_SAMPLES, 模型: $MODEL_NAME) ==="

# 运行 hotpotqa
echo ""
echo "=== 运行 hotpotqa 基准测试 ==="
uv run python run_llm_compiler.py \
    --benchmark_name hotpotqa \
    --store results/test_hotpotqa.json \
    --stream \
    --do_benchmark \
    --model_name "$MODEL_NAME" \
    --N $N_SAMPLES

# 运行 movie
echo ""
echo "=== 运行 movie 基准测试 ==="
uv run python run_llm_compiler.py \
    --benchmark_name movie \
    --store results/test_movie.json \
    --stream \
    --do_benchmark \
    --model_name "$MODEL_NAME" \
    --N $N_SAMPLES

# 运行 parallelqa
echo ""
echo "=== 运行 parallelqa 基准测试 ==="
uv run python run_llm_compiler.py \
    --benchmark_name parallelqa \
    --store results/test_parallelqa.json \
    --stream \
    --do_benchmark \
    --model_name "$MODEL_NAME" \
    --N $N_SAMPLES

echo ""
echo "=== 数据收集完成 ==="
echo "结果文件:"
ls -la results/test_*.json
