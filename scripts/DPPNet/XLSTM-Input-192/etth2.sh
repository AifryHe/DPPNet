export CUDA_VISIBLE_DEVICES=0
model_name=DPPNet

root_path_name=./dataset/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

model_type='XLSTM'
seq_len=192
for pred_len in 96 192 336 720
do
for random_seed in 2024 2025 2026 2027 2028
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 24 \
      --model_type $model_type \
      --train_epochs 100 \
      --patience 7 \
      --itr 1 --batch_size 256 --learning_rate 0.0002 --random_seed $random_seed
done
done


