#!/usr/bin/bash

set -x
umask 007
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

NGPU=${NGPU:-"8"}
MASTER_PORT=${MASTER_PORT:-"29501"}
PORT=${PORT:-"1106"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train"}
BASE_SAVE_ROOT=${SAVE_ROOT:-"/share/project/caomingyu/WAM_baseline/lingbot_va_results"}
FINAL_SAVE_ROOT="${BASE_SAVE_ROOT}/${TIMESTAMP}"

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}


overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

export WANDB_API_KEY="your key"
export WANDB_BASE_URL="your url"
export WANDB_TEAM_NAME="your team name"
export WANDB_PROJECT="your project"

## node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}
save_root=${FINAL_SAVE_ROOT}
## cmd setting
master_addr=${MASTER_ADDR}

export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_100,mlx5_101,mlx5_102,mlx5_103,mlx5_104,mlx5_105,mlx5_106,mlx5_107
export NCCL_IB_GID_INDEX=7
export NCCL_DEBUG=INFO
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master-port ${master_port} \
    --master_addr=${master_addr} \
    --tee 3 \
    -m wan_va.train --config-name ${config_name} \
    --save-root ${save_root} \
    $overrides
