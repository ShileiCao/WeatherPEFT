# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

TRAIN_START_DATE='2007' 
TRAIN_END_DATE='2017'

VAL_START_DATE="2017"
VAL_END_DATE="2019"


OUTPUT_DIR="output/WeatherPEFT" 

OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=8 \
    --master_port 12326 --nnodes=1  --node_rank=0 --master_addr="127.0.0.1" \
    run_downscaling.py \
    --log_dir ${OUTPUT_DIR} \
    --train_start_date ${TRAIN_START_DATE} \
    --train_end_date ${TRAIN_END_DATE} \
    --val_start_date ${VAL_START_DATE} \
    --val_end_date ${VAL_END_DATE} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 5 \
    --val_batch_size 30 \
    --save_ckpt_freq 10 \
    --opt adamw \
    --lr 7e-4 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --warmup_epochs 3 \
    --epochs 30 \
    --dist_eval \
    --mode ours \
    # --eval 


