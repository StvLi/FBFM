#!/bin/bash

# =============================================================================
# WAM Baseline 2节点分布式训练编排脚本
# =============================================================================

# 强制指定为绝对路径
SCRIPT_DIR="/share/project/caomingyu/WAM_baseline/lingbot-va"

# 统一生成时间戳，确保多机日志对齐
export SHARED_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "========================================================"
echo "本次多机训练的统一时间戳为: ${SHARED_TIMESTAMP}"
echo "========================================================"

# ============================ 可配置区域 ============================
# TODO: 每次重新申请机器后，请修改这里的 job ID
base_prefix="job-821c2a06-14a8-4617-be55-9411c6d095ea" 

TRAIN_SCRIPT="${SCRIPT_DIR}/script/run_va_posttrain_a2d.sh"

ENV_ACTIVATE='eval "$(conda shell.bash hook)" && conda activate /share/project/caomingyu/WAM_baseline/envs/lingbot-va/'

MASTER_PORT=29501
GPUS_PER_NODE=8
NUM_MACHINES=2

# 期望节点数
hosts=()
hosts+=("${base_prefix}-master-0")
hosts+=("${base_prefix}-worker-0")
# ====================================================================

idx_list=($(seq 0 $((${#hosts[@]}-1))))
MASTER_HOST=${hosts[${idx_list[0]}]}

echo "配置: ${NUM_MACHINES} 节点, 每节点 ${GPUS_PER_NODE} GPU, 主节点=${MASTER_HOST}, 端口=${MASTER_PORT}"

if [ ! -f "$TRAIN_SCRIPT" ]; then
  echo "错误: 训练脚本不存在: $TRAIN_SCRIPT"; exit 1
fi

run_distributed() {
  local TARGET_SCRIPT="$1"

  echo "1. 清理全集群遗留的 wan_va.train 进程..."
  for idx in "${idx_list[@]}"; do
    host=${hosts[$idx]}
    if [[ "$host" == "$(hostname)" ]]; then
      pkill -9 -f "wan_va.train" 2>/dev/null || true; sleep 1
    else
      if ssh -o ConnectTimeout=5 -o BatchMode=yes ${host} "echo ok" >/dev/null 2>&1; then
        ssh ${host} "pkill -9 -f 'wan_va.train' 2>/dev/null || true; sleep 1" || true
      else
        echo "警告: 无法连接到 ${host} 进行清理"
      fi
    fi
  done

  echo "2. 开始下发环境变量并拉起训练进程..."
  pids=(); ssh_failed=0; NODE_RANK=0
  for idx in "${idx_list[@]}"; do
    host=${hosts[$idx]}
    
    # 组装环境变量
    env_vars="export MASTER_ADDR=${MASTER_HOST}; export MASTER_PORT=${MASTER_PORT}; export NODE_RANK=${NODE_RANK}; export NUM_MACHINES=${NUM_MACHINES}; export GPUS_PER_NODE=${GPUS_PER_NODE}; export TIMESTAMP=${SHARED_TIMESTAMP};"
    
    # 组装完整执行命令：进目录 -> 激活Conda -> 导变量 -> 跑脚本
    RUN_CMD="cd ${SCRIPT_DIR}; ${ENV_ACTIVATE}; ${env_vars} bash ${TARGET_SCRIPT}"
    
    echo "  -> 启动: ${host} (RANK=${NODE_RANK})"
    if [[ "$host" == "$(hostname)" ]]; then
      (
        eval "${RUN_CMD}"
      ) &
      pids+=($!)
    else
      if ssh -o ConnectTimeout=5 -o BatchMode=yes ${host} "echo ok" >/dev/null 2>&1; then
        ssh ${host} "${RUN_CMD}" &
        pids+=($!)
      else
        echo "错误: 无法通过SSH连接到 ${host}"; ssh_failed=1
      fi
    fi
    NODE_RANK=$((NODE_RANK+1))
  done

  if [ $ssh_failed -eq 1 ]; then
    echo "致命错误: SSH连接失败"; exit 1
  fi

  echo "========================================================"
  echo "集群任务已全部下发！"
  echo "========================================================"
  
  failed=0
  if [ ${#pids[@]} -eq 0 ]; then echo "错误: 没有成功启动任何任务"; exit 1; fi
  for pid in "${pids[@]}"; do
    if ! wait $pid; then echo "任务 ${pid} 失败退出"; failed=1; fi
  done
  if [ $failed -ne 0 ]; then echo "错误: 训练异常中断"; exit 1; fi
}

run_distributed "$TRAIN_SCRIPT"