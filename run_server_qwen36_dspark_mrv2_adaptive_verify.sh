#!/usr/bin/env bash

# ============================================================
# 参数说明:
#   $1: 卡 ID (例如: 4,5,6,7 或 0,1,2,3，默认: 4,5,6,7)
#   $2: 端口号 (例如: 47008 或 45008，默认: 47008)
#   $3: 动态校验开关 (可选: av / none / true / false，默认: av)
#   $4: 投机 Token 数 (默认: 7)
#   $5: 投机推理方法 (可选: dspark / mtp，默认: dspark)
#   $6: Draft 采样方法是否使用 probabilistic (可选: true / false，默认: true)
#   $7: Profiling 路径 (可选，若不传或留空则不开启 profiler)
#
# 使用示例:
#   # 1. DSpark + 动态校验 + probabilistic
#   bash run_server.sh 4,5,6,7 47008 av 7 dspark true
#
#   # 2. DSpark + 关闭动态校验 + greedy
#   bash run_server.sh 4,5,6,7 47008 none 7 dspark false
#
#   # 3. MTP-7 + probabilistic
#   bash run_server.sh 4,5,6,7 47008 none 7 mtp true
#
#   # 4. MTP-3 + greedy
#   bash run_server.sh 4,5,6,7 47008 none 3 mtp false
# ============================================================

source /opt/cann-9.2.1/set_env.sh

DEVICES="${1:-4,5,6,7}"
PORT="${2:-47008}"
AV_MODE="${3:-av}"
NUM_SPECULATIVE_TOKENS="${4:-7}"
SPEC_METHOD="${5:-dspark}"
DRAFT_SAMPLE_PROBABILISTIC="${6:-true}"
PROFILER_DIR="${7:-}"
MAX_NUM_SEQS=64

# 校验投机 Token 数
[[ "${NUM_SPECULATIVE_TOKENS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "Error: num_speculative_tokens must be a positive integer"
    exit 1
}

# 校验投机推理方法
SPEC_METHOD="${SPEC_METHOD,,}"

case "${SPEC_METHOD}" in
    dspark|mtp)
        ;;
    *)
        echo "Error: Unknown speculative method '${SPEC_METHOD}'. Use 'dspark' or 'mtp'."
        exit 1
        ;;
esac

# 解析 Draft 采样方法开关
DRAFT_SAMPLE_PROBABILISTIC="${DRAFT_SAMPLE_PROBABILISTIC,,}"
case "${DRAFT_SAMPLE_PROBABILISTIC}" in
    true|1|yes)
        DRAFT_SAMPLE_METHOD="probabilistic"
        ;;
    false|0|no)
        DRAFT_SAMPLE_METHOD="greedy"
        ;;
    *)
        echo "Error: Unknown draft sample probabilistic flag '${DRAFT_SAMPLE_PROBABILISTIC}'. Use 'true' or 'false'."
        exit 1
        ;;
esac

# Capture Size：投机步数 + 1 的倍数
CAPTURE_STEP=$((NUM_SPECULATIVE_TOKENS + 1))
MAX_CAPTURE_SIZE=$((MAX_NUM_SEQS * CAPTURE_STEP))
CAPTURE_SIZES=$(seq -s, "${CAPTURE_STEP}" "${CAPTURE_STEP}" "${MAX_CAPTURE_SIZE}")

# 解析动态校验开关
case "${AV_MODE,,}" in
    av|true|1)
        ENABLE_AV="true"
        AV_TAG="av"
        ;;
    none|false|0)
        ENABLE_AV="false"
        AV_TAG="none"
        ;;
    *)
        echo "Error: Unknown adaptive verification mode '${AV_MODE}'. Use 'av'/'true' or 'none'/'false'."
        exit 1
        ;;
esac

echo "============================================================"
echo "Starting vLLM Server"
echo "Devices             : ${DEVICES}"
echo "Port                : ${PORT}"
echo "Spec Method         : ${SPEC_METHOD}"
echo "Spec Tokens         : ${NUM_SPECULATIVE_TOKENS}"
echo "Adaptive-V          : ${ENABLE_AV} (${AV_TAG})"
echo "Draft Sample Method : ${DRAFT_SAMPLE_METHOD}"

if [[ -n "${PROFILER_DIR}" ]]; then
    echo "Profiler            : Enabled -> ${PROFILER_DIR}"
else
    echo "Profiler            : Disabled"
fi

echo "============================================================"

# 防止 set_env.bash 遇到未绑定变量报错
export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/omni_custom_transformer/bin/set_env.bash

export ASCEND_RT_VISIBLE_DEVICES="${DEVICES}"
export VLLM_ASCEND_ENABLE_DSPARK_FIA_SINK=2
export VLLM_USE_V2_MODEL_RUNNER=1

# only qwen36 dspark need this 
# export VLLM_ASCEND_KV_GROUP_MIN_SIZE=16

export MODEL="/dpc/hot/model/z00927893/Qwen3.8-27B/"
export D_MODEL="/dpc/hot/w00958190/dspark/newly_specforge/qwen3.8_dspark/1111_MAE-CN_qwen38_dspark/output/qwen3.6-27b-dspark-offline-step10290-export"
export VLLM_DISABLE_COMPILE_CACHE=1

LOG_FILE="${SPEC_METHOD}_${AV_TAG}_sample_${DRAFT_SAMPLE_METHOD}_port${PORT}.log"

# 根据投机推理方法构建配置
if [[ "${SPEC_METHOD}" == "dspark" ]]; then
    SPEC_CONFIG="{\"num_speculative_tokens\": ${NUM_SPECULATIVE_TOKENS},\"method\":\"dspark\",\"model\":\"${D_MODEL}\",\"enable_adaptive_verification\": ${ENABLE_AV}, \"draft_sample_method\":\"${DRAFT_SAMPLE_METHOD}\"}"
else
    SPEC_CONFIG="{\"num_speculative_tokens\": ${NUM_SPECULATIVE_TOKENS},\"method\":\"mtp\", \"draft_sample_method\":\"${DRAFT_SAMPLE_METHOD}\"}"
fi

# 构建启动命令参数数组
cmd=(
    python -m vllm.entrypoints.openai.api_server
    --model "${MODEL}"
    --port "${PORT}"
    --data-parallel-size 1
    --tensor-parallel-size 4
    --gpu-memory-utilization 0.96
    --trust-remote-code
    --served-model-name qwen
    --max-model-len 65536
    --max-num-batched-tokens 4096
    --max-num-seqs "${MAX_NUM_SEQS}"
    --host "0.0.0.0"
    --block-size 128
    --tokenizer-mode auto
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --reasoning-parser qwen3
    --async-scheduling
    --enable-log-requests
    --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[${CAPTURE_SIZES}],\"max_cudagraph_capture_size\":${MAX_CAPTURE_SIZE}}"
    --speculative-config "${SPEC_CONFIG}"
)

# 如果指定了 profiling 路径，动态添加参数并创建目录
if [[ -n "${PROFILER_DIR}" ]]; then
    mkdir -p "${PROFILER_DIR}"
    cmd+=(--profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"${PROFILER_DIR}\", \"torch_profiler_with_stack\": true}")
fi

"${cmd[@]}" 2>&1 | tee "${LOG_FILE}"