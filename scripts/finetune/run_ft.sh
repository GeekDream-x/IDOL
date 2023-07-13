PROJECT_ROOT_DIR=${YOUR_WORK_DIR}
DATA_DIR=${PROJECT_ROOT_DIR}/data
model_name_or_path=${PROJECT_ROOT_DIR}/model
OUTPUT_DIR=${PROJECT_ROOT_DIR}/outputs


max_len=256
epoch=3.0
seed=31430
model_type=roberta
task_name=reclor # logiqa race
warmup_proportion=0.1   # 0.06 0.1
logging_steps=200 
bs=12
acc=2
lr=2e-5
weight_decay=0.0
adam_betas=(0.9,0.999)


mkdir -p $OUTPUT_DIR

python finetuning.py --model_type ${model_type} \
    --model_name_or_path ${model_name_or_path} \
    --task_name ${task_name} \
    --do_train \
    --evaluate_during_training \
    --do_test \
    --do_lower_case \
    --data_dir ${DATA_DIR} \
    --max_seq_length ${max_len} \
    --per_gpu_eval_batch_size ${bs} \
    --per_gpu_train_batch_size ${bs} \
    --gradient_accumulation_steps ${acc} \
    --learning_rate ${lr} \
    --num_train_epochs ${epoch} \
    --output_dir ${OUTPUT_DIR} \
    --logging_steps ${logging_steps} \
    --save_steps 200000 \
    --adam_epsilon 1e-6 \
    --no_clip_grad_norm \
    --warmup_proportion ${warmup_proportion} \
    --overwrite_output_dir \
    --weight_decay ${weight_decay} \
    --seed ${seed} \
    --adam_betas ${adam_betas} \
    --overwrite_cache \
    --fp16