#!/bin/bash

# =============================================================================
# WAM Baseline 一键清理所有分布式节点上的训练进程
# =============================================================================

# ============================ 可配置区域 ============================
# TODO: 每次重新申请机器后，请修改这里的 job ID
base_prefix="job-821c2a06-14a8-4617-be55-9411c6d095ea" 

# 期望节点数
hosts=()
hosts+=("${base_prefix}-master-0")
hosts+=("${base_prefix}-worker-0")
# ====================================================================

PROCESS_PATTERN="wan_va.train"

echo "========================================================"
echo "开始清理所有节点上的训练进程: ${PROCESS_PATTERN}"
echo "========================================================"

idx_list=($(seq 0 $((${#hosts[@]}-1))))
killed_count=0
not_found_count=0
failed_count=0

for idx in "${idx_list[@]}"; do
  host=${hosts[$idx]}
  
  if [[ "$host" == "$(hostname)" ]]; then
    # 本地节点
    echo "[本地] ${host}: 检查进程..."
    if pgrep -f "${PROCESS_PATTERN}" >/dev/null 2>&1; then
      pkill -9 -f "${PROCESS_PATTERN}" 2>/dev/null
      killed_count=$((killed_count + 1))
      echo "  ✓ 已清理"
    else
      not_found_count=$((not_found_count + 1))
      echo "  - 未找到运行中的进程"
    fi
    sleep 1
  else
    # 远程节点
    echo "[远程] ${host}: 检查进程..."
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes ${host} "echo ok" >/dev/null 2>&1; then
      echo "  ✗ 无法连接"
      failed_count=$((failed_count + 1))
      continue
    fi
    
    # 先检查进程是否存在
    if ssh ${host} "pgrep -f '${PROCESS_PATTERN}' >/dev/null 2>&1"; then
      ssh ${host} "pkill -9 -f '${PROCESS_PATTERN}' 2>/dev/null; sleep 1"
      killed_count=$((killed_count + 1))
      echo "  ✓ 已清理"
    else
      not_found_count=$((not_found_count + 1))
      echo "  - 未找到运行中的进程"
    fi
  fi
done

echo "========================================================"
echo "清理完成: 已清理 ${killed_count} 个节点, 未找到 ${not_found_count} 个节点, 连接失败 ${failed_count} 个节点"
echo "========================================================"

if [ $failed_count -gt 0 ]; then
  exit 1
fi