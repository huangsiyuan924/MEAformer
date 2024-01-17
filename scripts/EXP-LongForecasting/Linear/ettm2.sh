cd ../../../
if [ ! -d "logs" ]; then
  mkdir ./logs
fi

if [ ! -d "logs/LongForecasting" ]; then
  mkdir logs/LongForecasting
fi
seq_len=96
model_name=DLinear
for preLen in 24 48 96 288 672; do
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2_$seq_len'_'$preLen \
    --model $model_name \
    --data ETTm2 \
    --features M \
    --seq_len $seq_len \
    --pred_len $preLen \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 --batch_size 32 --learning_rate 0.001 >logs/LongForecasting/$model_name'_'ETTm2_$seq_len'_'$preLen.log
done
