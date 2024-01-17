cd ../../../../
if [ ! -d "logs" ]; then
  mkdir logs
fi

if [ ! -d "logs/LongForecasting" ]; then
  mkdir logs/LongForecasting
fi

if [ ! -d "logs/LongForecasting/univariate" ]; then
  mkdir logs/LongForecasting/univariate
fi
model_name=MyModel
input_len=96
for preLen in 96 192 336 720; do
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_336_96 \
    --model $model_name \
    --data ETTh2 \
    --seq_len $input_len \
    --pred_len $preLen \
    --enc_in 1 \
    --des 'Exp' \
    --itr 1 --batch_size 32 --feature S --learning_rate 0.005 >logs/LongForecasting/univariate/$model_name'_'fS_ETTh2_$input_len'_'$preLen.log
done