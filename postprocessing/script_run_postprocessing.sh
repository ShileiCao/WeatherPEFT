TRAIN_START_DATE='1998-01-01'
TRAIN_END_DATE='2015-12-31'

VAL_START_DATE="2016-01-03"
VAL_END_DATE="2017-12-31"


OUTPUT_DIR="output/WeatherPEFT"
OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=8 \
    --master_port 12376 --nnodes=1  --node_rank=0 --master_addr="127.0.0.1" \
    run_postprocessing.py \
    --mode ours \
    --log_dir ${OUTPUT_DIR} \
    --train_start_date ${TRAIN_START_DATE} \
    --train_end_date ${TRAIN_END_DATE} \
    --val_start_date ${VAL_START_DATE} \
    --val_end_date ${VAL_END_DATE} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --val_batch_size 4 \
    --save_ckpt_freq 5 \
    --opt adamw \
    --lr 1e-3 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --warmup_epochs 3 \
    --epochs 10 \
    --dist_eval \
    --clip_grad 5 \
    # --eval \
