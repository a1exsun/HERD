export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

swift sft \
    --model $1 \
    --train_type full \
    --dataset $2 \
    --num_train_epochs $3 \
    --per_device_train_batch_size 16 \
    --learning_rate 1e-5 \
    --save_total_limit 1 \
    --max_length 2048 \
    --output_dir $4 \
    --warmup_ratio 0.0 \
    --weight_decay 0.0 \
    --lr_scheduler_type linear \
    --save_only_model true \
    --use_liger_kernel true \
    --columns '{"prompt": "query", "answer": "response"}' \
    --deepspeed zero2 