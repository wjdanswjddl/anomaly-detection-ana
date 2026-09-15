# Shared defaults for all diffusion training configs.
# Sourced by linear/cosine/ramp/anisotropic/pred_xstart.sh
#
# Architecture matches gputnam training-SBND/iterE/flags.txt.
# Batch is lowered for EAF MIG VRAM (iterE used batch 50 / microbatch 10 on a
# larger GPU). Raise BATCH_SIZE if nvidia-smi shows headroom.
#
# Intentional improvements vs bare iterE (wall-time / robustness):
#   validation_interval 500  (code default was 5 — huge overhead)
#   plot_interval 10000      (iterE used 2000; plots run full DDPM+DDIM)
#   use_fp16 True            (A100/MIG; iterE left False)
#   max_steps 111000         (stop near iterE's ~111k ckpt; constant LR)
#
# Override before sourcing, e.g.:
#   export DATA_DIR=/scratch/.../npz
#   export BATCH_SIZE=16
#   source configs/train_flags/linear.sh

: "${DATA_DIR:=/scratch/7DayExclusive/munjung/anomaly-detection/npz}"
: "${VAL_DIR:=${DATA_DIR}}"
# iterE flags.txt omitted charge_scale → argparse default 1.0.
: "${CHARGE_SCALE:=1}"
# MIG-safe defaults (peak VRAM ~ microbatch * 512^2 activations).
: "${BATCH_SIZE:=8}"
: "${MICROBATCH:=4}"

# Architecture — identical to iterE.
export MODEL_FLAGS="--image_size 512 --num_channels 32 --class_cond False --num_res_blocks 2 --num_heads 8 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16,32 --channel_mult 1,2,4,8,8,8"

# Optimizer / batch — MIG-safe sizes + wall-time / stop improvements.
export TRAIN_FLAGS="--lr 1e-4 --batch_size ${BATCH_SIZE} --microbatch ${MICROBATCH} --weight_batches False --weight_pixels False --weight_decay 0.01 --save_interval 1000 --plot_interval 10000 --validation_interval 500 --use_fp16 True --max_steps 111000 --lr_anneal_steps 0"

export DATADIR="--data_dir ${DATA_DIR} --validation_dir ${VAL_DIR} --charge_scale ${CHARGE_SCALE}"
