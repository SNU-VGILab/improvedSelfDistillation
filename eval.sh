CUDA_VISIBLE_DEVICES=2 accelerate launch \
    --dynamo_backend=no \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    eval.py \
    --targets 2026.01.18KST19.26.11-xlarge1 \
    --checkpoints 0600000.pt \
    --sampling-steps 2 \
    --precfg 5.0 \
    --cfg-start 0.0 \
    --cfg-end 1.0

CUDA_VISIBLE_DEVICES=2 accelerate launch \
    --dynamo_backend=no \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    eval.py \
    --targets 2026.02.15KST14.22.08-base4 \
    --checkpoints 0400000.pt \
    --sampling-steps 2
