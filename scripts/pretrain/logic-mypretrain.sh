
PROJECT_ROOT_DIR=${YOUR_WORK_DIR}
DATA_DIR=${PROJECT_ROOT_DIR}/data
LOG_DIR=${PROJECT_ROOT_DIR}/logs
model_name_or_path=${PROJECT_ROOT_DIR}/deberta-v2-xxlarge
OUTPUT_DIR=${PROJECT_ROOT_DIR}/outputs/${taskname}
data_cache_dir=${PROJECT_ROOT_DIR}/data_cache_dir

batch_size=1
accu=2
ngpu=8
lr=2e-5
lcp_weight=0.8
train_steps=25000
warmup_steps=2500
logging_steps=250
save_steps=2500
max_seq_length=512
model_type=roberta
dataloader_num_workers=2


python -m torch.distributed.launch --nnodes \$WORLD_SIZE --node_rank \$RANK --master_addr \$MASTER_ADDR --master_port \$MASTER_PORT --nproc_per_node ${ngpu} \
    ../run.py \
    --config_name ${model_name_or_path} \
    --tokenizer_name ${model_name_or_path} \
    --model_name_or_path ${model_name_or_path} \
    --train_files_dir ${DATA_DIR} \
    --preprocessing_num_workers 20 \
    --max_seq_length ${max_seq_length} \
    --output_dir ${OUTPUT_DIR} \
    --do_train \
    --max_steps ${train_steps} \
    --dataloader_num_workers ${dataloader_num_workers} \
    --per_device_train_batch_size ${batch_size} \
    --learning_rate ${lr} \
    --warmup_steps ${warmup_steps} \
    --logging_steps ${logging_steps} \
    --save_strategy steps \
    --save_steps ${save_steps} \
    --save_total_limit 500 \
    --seed $torch_seed \
    --fp16 \
    --gradient_accumulation_steps ${accu} \
    --data_cache_dir ${data_cache_dir} \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --max_grad_norm 0 \
    --weight_decay 0.01 \
    --lcp_weight ${lcp_weight} \
    --overwrite_output_dir \
    --model_type ${model_type}
 