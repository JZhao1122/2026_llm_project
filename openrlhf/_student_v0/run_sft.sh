export CUDA_VISIBLE_DEVICES=0,1
cd /root/workspace/repos_zhaojian/2026_llm_project
deepspeed --module openrlhf.cli.train_sft \
   --pretrain /root/workspace/_hf_models/Qwen/Qwen2.5-1.5B \
   --dataset openai/gsm8k \
   --dataset_split train \
   --input_key question \
   --output_key answer \
   --max_len 2048 \
   --max_epochs 3 \
   --train_batch_size 128 \
   --micro_train_batch_size 8 \
   --learning_rate 5e-6 \
   --lr_scheduler cosine_with_min_lr \
   --lr_warmup_ratio 0.03 \
   --save_path ./ckpt/qwen2.5-1.5b-sft \
   --zero_stage 2 \
   --param_dtype bf16 \
   --gradient_checkpointing \
   --attn_implementation flash_attention_2 \
   --logging_steps 10 \
   --save_steps -1 \
   --eval_steps -1
