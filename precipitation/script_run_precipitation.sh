TRAIN_START_DATE='2010'
TRAIN_END_DATE='2020'

VAL_START_DATE="2020"
VAL_END_DATE="2021"


MODEL='ours'

NOISEWEIGHT='0.2'


OUTPUT_DIR="output/WeatherPEFT"
OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=8 \
    --master_port 12325 --nnodes=1  --node_rank=0 --master_addr="127.0.0.1" \
    run_precipitation.py \
    --mode ${MODEL} \
    --log_dir ${OUTPUT_DIR} \
    --train_start_date ${TRAIN_START_DATE} \
    --train_end_date ${TRAIN_END_DATE} \
    --val_start_date ${VAL_START_DATE} \
    --val_end_date ${VAL_END_DATE} \
    --output_dir ${OUTPUT_DIR} \
    --horizon 12 \
    --batch_size 4 \
    --val_batch_size 16 \
    --save_ckpt_freq 4 \
    --opt adamw \
    --lr 3e-3 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --warmup_epochs 3 \
    --epochs 15 \
    --dist_eval \
    --noise_weight ${NOISEWEIGHT} \
    --k_value 0.001
    # --eval \
