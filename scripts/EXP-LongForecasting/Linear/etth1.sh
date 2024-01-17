cd ../../../
if [ ! -d "logs" ]; then
    mkdir ./logs
fi

if [ ! -d "logs/LongForecasting" ]; then
    mkdir logs/LongForecasting
fi
seq_len=96
model_name=DLinear

for preLen in 24 48 168 336 720; do

  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_$seq_len'_'$preLen \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --pred_len $preLen \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 --batch_size 32 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Etth1_$seq_len'_'$preLen.log
done
#python -u run_longExp.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_$seq_len'_'48 \
#  --model $model_name \
#  --data ETTh1 \
#  --features M \
#  --seq_len $seq_len \
#  --pred_len 48 \
#  --enc_in 7 \
#  --des 'Exp' \
#  --itr 1 --batch_size 32 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Etth1_$seq_len'_'96.log
#
#python -u run_longExp.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_$seq_len'_'168 \
#  --model $model_name \
#  --data ETTh1 \
#  --features M \
#  --seq_len $seq_len \
#  --pred_len 168 \
#  --enc_in 7 \
#  --des 'Exp' \
#  --itr 1 --batch_size 32 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Etth1_$seq_len'_'96.log
#
#python -u run_longExp.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_$seq_len'_'336 \
#  --model $model_name \
#  --data ETTh1 \
#  --features M \
#  --seq_len $seq_len \
#  --pred_len 336 \
#  --enc_in 7 \
#  --des 'Exp' \
#  --itr 1 --batch_size 32 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Etth1_$seq_len'_'192.log
#
#python -u run_longExp.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_$seq_len'_'720 \
#  --model $model_name \
#  --data ETTh1 \
#  --features M \
#  --seq_len $seq_len \
#  --pred_len 720 \
#  --enc_in 7 \
#  --des 'Exp' \
#  --itr 1 --batch_size 32 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Etth1_$seq_len'_'720.log