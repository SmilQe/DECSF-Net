#!/bin/bash

# 创建日志目录
mkdir -p ./logs_imm/LongForecasting

# 设置变量
water_seq_len=336
mete_seq_len=672
label_len=48
model_name="iMMTST"     # iMMTST 就是 DECSF-Net
root_path_name="./dataset/"
data_path_name="ffill.csv"
model_id_name="iMMTST" # iMMTST 就是 DECSF-Net
data_name="MultiModel"
random_seed=2021

# 如果遍历不同长度
#for pred_len in 96 192 336 720;

# 这里只设置 pred_len = 96
for pred_len in 96; do
    log_file="./logs_imm/LongForecasting/${model_name}_${data_path_name}_${model_id_name}_${water_seq_len}_${mete_seq_len}_${label_len}_${pred_len}.log"

    echo "Running pred_len=${pred_len}..."

    python -u run_longExp_imm.py \
        --random_seed ${random_seed} \
        --is_training 1 \
        --root_path ${root_path_name} \
        --data_path ${data_path_name} \
        --model_id ${model_id_name}_${water_seq_len}_${mete_seq_len}_${label_len}_${pred_len} \
        --model ${model_name} \
        --data ${data_name} \
        --features M \
        --water_seq_len ${water_seq_len} \
        --mete_seq_len ${mete_seq_len} \
        --label_len ${label_len} \
        --pred_len ${pred_len} \
        --water_enc_in 4 \
        --mete_enc_in 6 \
        --water_dec_in 4 \
        --c_out 4 \
        --d_model 512 \
        --n_heads 16 \
        --ew_layers 1 \
        --em_layers 1 \
        --dw_layers 1 \
        --b_layers 1 \
        --ba_layers 1 \
        --d_ff 512 \
        --dropout 0.1 \
        --embed timeF \
        --activation gelu \
        --num_workers 8 \
        --itr 1 \
        --train_epochs 100 \
        --batch_size 128 \
        --patience 10 \
        --learning_rate 0.0001 \
        --devices 0,1 \
        --use_amp \
    2>&1 | tee -a ${log_file}
done

