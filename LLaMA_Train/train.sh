export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_DISABLED=true
torchrun --nproc_per_node=4 --master_port=30500 alpaca_train.py \
    --model_name_or_path ./LLaMA-7B \
    --train_data_path ../datasets/llama_train/all_train.jsonl \
    --eval_data_path ../datasets/llama_train/all_eval.jsonl \
    --bf16 True \
    --tf32 True \
    --do_train \
    --do_eval \
    --output_dir ./LLaMA-Toolink \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy epoch \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 50 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'
