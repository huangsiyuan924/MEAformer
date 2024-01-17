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
    --data_path ETTh1.csv \
    --model_id ETTh1_336_96 \
    --model $model_name \
    --data ETTh1 \
    --seq_len $input_len \
    --pred_len $preLen \
    --enc_in 1 \
    --des 'Exp' \
    --itr 1 --batch_size 32 --feature S --learning_rate 0.005 >logs/LongForecasting/univariate/$model_name'_'fS_ETTh1_$input_len'_'$preLen.log
done
#python -u run_longExp.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_336_192 \
#  --model $model_name \
#  --data ETTh1 \
#  --seq_len 336 \
#  --pred_len 192 \
#  --enc_in 1 \
#  --des 'Exp' \
#  --itr 1 --batch_size 32 --feature S --learning_rate 0.005 >logs/LongForecasting/$model_name'_'fS_ETTh1_336_192.log
#
#python -u run_longExp.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_336_336 \
#  --model $model_name \
#  --data ETTh1 \
#  --seq_len 336 \
#  --pred_len 336 \
#  --enc_in 1 \
#  --des 'Exp' \
#  --itr 1 --batch_size 32 --feature S --learning_rate 0.005 >logs/LongForecasting/$model_name'_'fS_ETTh1_336_336.log
#
#
#python -u run_longExp.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_336_720 \
#  --model $model_name \
#  --data ETTh1 \
#  --seq_len 336 \
#  --pred_len 720 \
#  --enc_in 1 \
#  --des 'Exp' \
#  --itr 1 --batch_size 32 --feature S --learning_rate 0.005 >logs/LongForecasting/$model_name'_'fS_ETTh1_336_720.log
#
