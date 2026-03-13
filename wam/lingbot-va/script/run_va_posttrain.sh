#!/usr/bin/bash

set -x
umask 007

# 1. 接收外层传进来的环境变量
TIMESTAMP=${TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}
NNODES=${NUM_MACHINES:-"1"}           
NGPU=${GPUS_PER_NODE:-"8"}            
NODE_RANK=${NODE_RANK:-"0"}
MASTER_PORT=${MASTER_PORT:-"29501"}

if [ "${NNODES}" -gt 1 ] && [ -z "${MASTER_ADDR}" ]; then
    echo "【致命错误】：检测到多节点训练 (NNODES=${NNODES})，但外层没有传入 MASTER_ADDR 参数！"
    echo "为防止 NCCL 通信死锁，已主动阻断程序。请检查启动编排脚本。"
    exit 1
fi
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# ===================================================================

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train"}

BASE_SAVE_ROOT=${SAVE_ROOT:-"/share/project/caomingyu/WAM_baseline/lingbot_va_results"}
FINAL_SAVE_ROOT="${BASE_SAVE_ROOT}/${TIMESTAMP}"

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

export WANDB_API_KEY="your key"
export WANDB_BASE_URL="your url"
export WANDB_TEAM_NAME="your team name"
export WANDB_PROJECT="your project"

## cmd setting
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_100,mlx5_101,mlx5_102,mlx5_103,mlx5_104,mlx5_105,mlx5_106,mlx5_107
export NCCL_IB_GID_INDEX=7
export NCCL_DEBUG=INFO

ENV_PYTHON="/share/project/caomingyu/WAM_baseline/envs/lingbot-va/bin/python"

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
${ENV_PYTHON} -m torch.distributed.run \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --nproc_per_node=${NGPU} \
    --master-port ${MASTER_PORT} \
    --master_addr=${MASTER_ADDR} \
    --tee 3 \
    -m wan_va.train --config-name ${CONFIG_NAME} \
    --save-root ${FINAL_SAVE_ROOT} \
    $overrides