if [ ! -d "logs" ]; then
    mkdir logs
fi

if [ ! -d "logs/LongForecasting" ]; then
    mkdir logs/LongForecasting
fi

if [ ! -d "logs/LongForecasting/univariate1" ]; then
    mkdir logs/LongForecasting/univariate1
fi
input_len=96
model_name=MEAformer
e_layers=2
d_layers=1
d_ff=512
S=256
# ETTm2, univariate results, pred_len= 24 48 96 192 336 720
#python -u run_longExp.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm2.csv \
#  --model_id ETTm2_96_96 \
#  --model $model_name \
#  --data ETTm2 \
#  --seq_len 96 \
#  --pred_len 96 \
#  --enc_in 1 \
#  --des 'Exp' \
#  --e_layers $e_layers \
#  --d_layers $d_layers \
#  --d_ff $d_ff \
#  --S $S \
#  --itr 1 --batch_size 32 --learning_rate 0.001 --feature S >logs/LongForecasting/univariate/$model_name'_'fS_ETTm2_96_96'_'el$e_layers'_'dl$d_layers'_'d_ff$d_ff'_'S$S.log

#python -u run_longExp.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm2.csv \
#  --model_id ETTm2_96_192 \
#  --model $model_name \
#  --data ETTm2 \
#  --seq_len 96 \
#  --pred_len 192 \
#  --enc_in 1 \
#  --des 'Exp' \
#  --e_layers $e_layers \
#  --d_layers $d_layers \
#  --d_ff $d_ff \
#  --S $S \
#  --itr 1 --batch_size 32 --learning_rate 0.001 --feature S >logs/LongForecasting/univariate/$model_name'_'fS_ETTm2_96_192'_'el$e_layers'_'dl$d_layers'_'d_ff$d_ff'_'S$S.log
#
#python -u run_longExp.py \
#  --is_training 1 \
#  --root_path ./dataset/ETT-small/ \
#  --data_path ETTm2.csv \
#  --model_id ETTm2_96_336 \
#  --model $model_name \
#  --data ETTm2 \
#  --seq_len 96 \
#  --pred_len 336 \
#  --enc_in 1 \
#  --des 'Exp' \
#  --e_layers $e_layers \
#  --d_layers $d_layers \
#  --d_ff $d_ff \
#  --S $S \
#  --itr 1 --batch_size 32 --learning_rate 0.001 --feature S >logs/LongForecasting/univariate/$model_name'_'fS_ETTm2_96_336'_'el$e_layers'_'dl$d_layers'_'d_ff$d_ff'_'S$S.log
#
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_720 \
  --model $model_name \
  --data ETTm2 \
  --seq_len 96 \
  --pred_len 720 \
  --enc_in 1 \
  --des 'Exp' \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --d_ff $d_ff \
  --S $S \
  --itr 1 --batch_size 32 --learning_rate 0.001 --feature S >logs/LongForecasting/univariate1/$model_name'_'fS_ETTm2_96_720'_'el$e_layers'_'dl$d_layers'_'d_ff$d_ff'_'S$S.log