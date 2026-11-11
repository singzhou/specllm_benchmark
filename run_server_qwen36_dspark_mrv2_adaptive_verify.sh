#!/usr/bin/env bash

# ============================================================
# 参数说明:
#   $1: 卡 ID (例如: 4,5,6,7 或 0,1,2,3，默认: 4,5,6,7)
#   $2: 端口号 (例如: 47008 或 45008，默认: 47008)
#   $3: 动态校验开关 (可选: av / none / true / false，默认: av)
#   $4: Profiling 路径 (可选，若不传或留空则不开启 profiler)
#
# 使用示例:
#   # 1. 正常启动，不开启 profiling
#   bash run_server.sh 4,5,6,7 47008 av
#
#   # 2. 开启 profiling 并指定落盘路径
#   bash run_server.sh 4,5,6,7 47008 av /sharedstorage/z00830407/dspark_av_profiling_b4-1
# ============================================================

source /opt/cann-9.2.1/set_env.sh

DEVICES="${1:-4,5,6,7}"
PORT="${2:-47008}"
AV_MODE="${3:-av}"
NUM_SPECULATIVE_TOKENS="${4:-7}"
PROFILER_DIR="${5:-}"
MAX_NUM_SEQS=64

[[ "${NUM_SPECULATIVE_TOKENS}" =~ ^[1-9][0-9]*$ ]] || { echo "Error: num_speculative_tokens must be a positive integer"; exit 1; }

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
echo "Devices    : ${DEVICES}"
echo "Port       : ${PORT}"
echo "Adaptive-V : ${ENABLE_AV} (${AV_TAG})"
if [[ -n "${PROFILER_DIR}" ]]; then
    echo "Profiler   : Enabled -> ${PROFILER_DIR}"
else
    echo "Profiler   : Disabled"
fi
echo "============================================================"

# 防止 set_env.bash 遇到未绑定变量报错
export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/omni_custom_transformer/bin/set_env.bash

export ASCEND_RT_VISIBLE_DEVICES="${DEVICES}"
export VLLM_ASCEND_ENABLE_DSPARK_FIA_SINK=2
# export VLLM_ASCEND_ENABLE_MEGA_CHUNK_GDN=1
# export VLLM_ASCEND_MEGA_CHUNK_GDN_DEBUG=1
export VLLM_USE_V2_MODEL_RUNNER=1
# export VLLM_ASCEND_KV_GROUP_MIN_SIZE=16
# export VLLM_ASCEND_DECISIONS_WARM_SHARED_PREFIX=1

export MODEL="/dpc/hot/model/z00927893/Qwen3.6-27B/"
export D_MODEL="/dpc/cold/z00830407/DSpark_project/qwen36_training_data/turn2/model/qwen3.6-27b-dspark-naie-turn2"


# export MODEL="/dpc/hot/model/z00927893/Qwen3.8-27B/"
# export D_MODEL="/dpc/hot/w00958190/dspark/newly_specforge/qwen3.8_dspark/1020_domain_qwen38_dspark/output_1031_domain_noctx/qwen3.6-27b-dspark-offline-latest-export"

# export D_MODEL="/dpc/hot/w00958190/dspark/newly_specforge/qwen3.8_dspark/1020_domain_qwen38_dspark/output_ctx2048/1030_qwen38_ctx_80k+40k_check_point_step11232/"
# export D_MODEL="/dpc/hot/w00958190/dspark/newly_specforge/qwen3.8_dspark/1008_500k_qwen3.8_dspark/output_200k+200k/qwen3.6-27b-dspark-offline-latest-export
# export D_MODEL="/dpc/hot/w00958190/dspark/newly_specforge/qwen3.8_dspark/1020_domain_qwen38_dspark/output_1031_domain_noctx/qwen3.6-27b-dspark-offline-latest-export"
# export MODEL="/dpc/hot/z00896882/output/jev-qwen35-27b-sft-checkpoint26-merged-hf"
# export D_MODEL="/dpc/cold/z00830407/dspark_model/qwen36_dspark_common"
export VLLM_DISABLE_COMPILE_CACHE=1

LOG_FILE="dspark_${AV_TAG}_port${PORT}.log"

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
    --speculative-config "{\"num_speculative_tokens\": ${NUM_SPECULATIVE_TOKENS},\"method\":\"dspark\",\"model\":\"${D_MODEL}\", \"enable_adaptive_verification\": ${ENABLE_AV}}"
)
    # --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":256}'
    # --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":768}'
    # --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":768}'
    # --enforce-eager
    # --quantization "ascend"
    # --logprobs-mode raw_logprobs

# 如果指定了 profiling 路径，动态添加参数并创建目录
if [[ -n "${PROFILER_DIR}" ]]; then
    mkdir -p "${PROFILER_DIR}"
    # cmd+=(--profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"${PROFILER_DIR}\"}")
    cmd+=(--profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"${PROFILER_DIR}\", \"torch_profiler_with_stack\": true}")
fi

"${cmd[@]}" 2>&1 | tee "${LOG_FILE}"