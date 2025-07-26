if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo ".env not found. STOP"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=5

accelerate launch --debug --config_file ./configs/accelerate1.yaml train.py