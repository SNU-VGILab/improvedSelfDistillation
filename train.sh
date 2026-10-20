CUDA_VISIBLE_DEVICES=1 accelerate launch \
    --dynamo_backend=no \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    train.py \
    --config ./configs/base4.yaml
