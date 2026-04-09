START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}

save_root='visualization/'
mkdir -p $save_root

if ! python -c "import importlib.util; import sys; required = ('torch', 'diffusers', 'transformers'); missing = [name for name in required if importlib.util.find_spec(name) is None]; sys.exit(0 if not missing else 1)"; then
    echo "launch_server.sh must be run in the LingBot-VA model environment." >&2
    echo "Missing one or more required Python packages: torch, diffusers, transformers." >&2
    echo "Activate the env that contains LingBot-VA inference dependencies, for example: conda activate lingbot-va" >&2
    exit 1
fi

python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port $MASTER_PORT \
    wan_va/wan_va_server.py \
    --config-name robotwin \
    --port $START_PORT \
    --save_root $save_root

