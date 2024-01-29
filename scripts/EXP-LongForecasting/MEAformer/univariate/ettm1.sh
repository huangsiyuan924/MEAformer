if [ ! -d "logs" ]; then
  mkdir logs
fi

if [ ! -d "logs/LongForecasting" ]; then
  mkdir logs/LongForecasting
fi

if [ ! -d "logs/LongForecasting/univariate" ]; then
  mkdir logs/LongForecasting/univariate
fi
input_len=96
model_name=MEAformer
e_layers=2
d_layers=1
d_ff=512
S=256
for preLen in 96 192 336 720; do
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id ETTm1_$input_len'_'$preLen \
    --model $model_name \
    --data ETTm1 \
    --seq_len $input_len \
    --pred_len $preLen \
    --enc_in 1 \
    --des 'Exp' \
    --e_layers $e_layers \
    --d_layers $d_layers \
    --d_ff $d_ff \
    --S $S \
    --itr 1 --batch_size 8 --feature S --learning_rate 0.0001 >logs/LongForecasting/univariate/$model_name'_'fS_ETTm1_$input_len'_'$preLen'_'el$e_layers'_'dl$d_layers'_'d_ff$d_ff'_'S$S.log
done