export NPROC_PER_NODE=1  # [HERO] one GPU, was 8
export CUDA_VISIBLE_DEVICES=0  # [HERO] was 0,1,2,3,4,5,6,7

# [HERO] edits to the command below:
# [HERO]   --train_type full      -> --tuner_type lora --lora_rank 32 --lora_alpha 64 --lora_dropout 0 (ms-swift 4 renamed
# [HERO]                             the flag; protocol LoRA budget), plus --adapters $5 to continue the previous stage's LoRA
# [HERO]   --num_train_epochs $3  -> --max_steps $3 (protocol: 32 updates per stage)
# [HERO]   --per_device_train_batch_size 16 -> 1 with --gradient_accumulation_steps 32 (protocol batch 32 on one GPU)
# [HERO]   --learning_rate 1e-5   -> 1e-4 (protocol LoRA learning rate)
# [HERO]   --max_length 2048      -> 16384 (protocol sequence ceiling)
# [HERO]   --lazy_tokenize false  added: drop over-length samples as ms-swift 3.12 did for a text-only model
# [HERO]   --model_type qwen3_5 --template qwen3_5  added: ms-swift 4.5.3 cannot infer either for a local Qwen3.5
# [HERO]                             checkpoint directory (several registrations match its architecture)
# [HERO]   --dataloader_num_workers 0  added: a worker process rebuilds the template's model from the directory
# [HERO]                             alone and hits the same model_type ambiguity; collating in the main process avoids it
# [HERO]   --dataloader_persistent_workers false  added: required with 0 workers (ms-swift 3.12's default; 4.5.3 defaults to true)
# [HERO]   --deepspeed zero2      removed (one GPU)
swift sft \
    --model $1 \
    --model_type qwen3_5 \
    --template qwen3_5 \
    --tuner_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0 \
    ${5:+--adapters $5} \
    --dataset $2 \
    --max_steps $3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-4 \
    --save_total_limit 1 \
    --max_length 16384 \
    --lazy_tokenize false \
    --dataloader_num_workers 0 \
    --dataloader_persistent_workers false \
    --output_dir $4 \
    --warmup_ratio 0.0 \
    --weight_decay 0.0 \
    --lr_scheduler_type linear \
    --save_only_model true \
    --use_liger_kernel true \
    --columns '{"prompt": "query", "answer": "response"}'  # [HERO] last flag: --deepspeed zero2 removed
