START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}

save_root='visualization/'
mkdir -p $save_root

route_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="${route_root}:${PYTHONPATH:-}"

"${LINGBOT_SERVER_PYTHON:-python}" -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port $MASTER_PORT \
    wan_va/wan_va_server.py \
    --config-name robotwin \
    --port $START_PORT \
    --save_root $save_root


