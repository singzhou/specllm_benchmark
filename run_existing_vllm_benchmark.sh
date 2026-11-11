#!/usr/bin/env bash
# 复用已启动的模型服务，顺序评测所有 final_answer / tool_call 数据集。
set -euo pipefail
shopt -s nullglob

# 按当前已运行的服务修改；也可在执行时通过同名环境变量覆盖。
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-43119}"
MODEL_NAME="${MODEL_NAME:-qwen}"  # 服务的 --served-model-name
TOKENIZER_PATH="${TOKENIZER_PATH:-/dpc/hot/model/z00927893/Qwen3.8-27B/}"
SPEC_STEPS="${SPEC_STEPS:-7}"  # 与当前服务一致；无投机时设为 0
CONCURRENCY="${CONCURRENCY:-8}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BENCH_SCRIPT="${BENCH_SCRIPT:-agent_benchmark.py}"
RESULT_DIR="${RESULT_DIR:-test_output/dspark-7-qwen36}"

FINAL_ANSWER_DATASETS=(
    ./testsets/common_200_openai.jsonl
)

FINAL_ANSWER_NOCOT_DATASETS=(
    ./testsets/common_200_openai.jsonl
)

TOOL_CALL_DATASETS=(
    ./testsets/MAE-CN-wenshu-toolcall.jsonl
)

TOOL_CALL_NOCOT_DATASETS=(
    ./testsets/MAE-CN-wenshu-toolcall.jsonl
)

[[ -f "$BENCH_SCRIPT" ]] || { echo "找不到评测脚本: $BENCH_SCRIPT" >&2; exit 1; }
if (( ${#FINAL_ANSWER_DATASETS[@]} + ${#TOOL_CALL_DATASETS[@]} == 0 )); then
    echo "未找到任何 JSONL 测试集，请检查上面的路径。" >&2
    exit 1
fi

run_stage() {
    local phase="$1"
    shift
    local dataset name
    mkdir -p "$RESULT_DIR/$phase/logs"
    for dataset in "$@"; do
        name="$(basename "$dataset" .jsonl)"
        echo "开始 [$phase] $name"
        "$PYTHON_BIN" -u "$BENCH_SCRIPT" \
            --dataset "$dataset" \
            --output_file "$RESULT_DIR/$phase/$name.jsonl" \
            --tokenizer_path "$TOKENIZER_PATH" \
            --ip "$HOST" --port "$PORT" --model "$MODEL_NAME" \
            --concurrency "$CONCURRENCY" --spec_step_num "$SPEC_STEPS" \
            --eval_target "$phase" --enable_thinking \
            --max_tokens "$MAX_TOKENS" --temperature "$TEMPERATURE" --top_p "$TOP_P" \
            --request_timeout "$REQUEST_TIMEOUT" \
            --metrics-by-category \
            2>&1 | tee "$RESULT_DIR/$phase/logs/$name.log"
    done
}

run_stage_nocot() {
    local phase="$1"
    shift
    local dataset name
    mkdir -p "$RESULT_DIR/$phase/logs"
    for dataset in "$@"; do
        name="$(basename "$dataset" .jsonl)"
        echo "开始 [$phase] $name"
        "$PYTHON_BIN" -u "$BENCH_SCRIPT" \
            --dataset "$dataset" \
            --output_file "$RESULT_DIR/$phase/nocot_$name.jsonl" \
            --tokenizer_path "$TOKENIZER_PATH" \
            --ip "$HOST" --port "$PORT" --model "$MODEL_NAME" \
            --concurrency "$CONCURRENCY" --spec_step_num "$SPEC_STEPS" \
            --eval_target "$phase" \
            --max_tokens "$MAX_TOKENS" --temperature "$TEMPERATURE" --top_p "$TOP_P" \
            --request_timeout "$REQUEST_TIMEOUT" \
            --metrics-by-category \
            2>&1 | tee "$RESULT_DIR/$phase/logs/$name.log"
    done
}

echo "服务: http://$HOST:$PORT | model=$MODEL_NAME | spec_steps=$SPEC_STEPS"
echo "数据集: ${#FINAL_ANSWER_DATASETS[@]} 个 final_answer + ${#TOOL_CALL_DATASETS[@]} 个 tool_call"
echo "结果目录: $RESULT_DIR"
run_stage_nocot final_answer "${FINAL_ANSWER_NOCOT_DATASETS[@]}"
# run_stage_nocot tool_call "${TOOL_CALL_NOCOT_DATASETS[@]}"
run_stage final_answer "${FINAL_ANSWER_DATASETS[@]}"
# run_stage tool_call "${TOOL_CALL_DATASETS[@]}"
echo "全部完成，结果和日志: $RESULT_DIR"
