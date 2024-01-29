if [ ! -d "logs" ]; then
    mkdir ./logs
fi

if [ ! -d "logs/LongForecasting_test" ]; then
    mkdir logs/LongForecasting_test
fi
seq_len=96
model_name=MEAformer
e_layers=2
d_layers=1
d_ff=512
S=512
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 8 \
  --des 'Exp' \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --d_ff $d_ff \
  --S $S \
  --itr 1 --batch_size 8 --learning_rate 0.0001 >logs/LongForecasting_test/$model_name'_'Exchange_$seq_len'_'96_el$e_layers'_'dl$d_layers'_'d_ff$d_ff'_'S$S.log

#python -u run_longExp.py \
#  --is_training 1 \
#  --root_path ./dataset/exchange_rate/ \
#  --data_path exchange_rate.csv \
#  --model_id Exchange_$seq_len'_'192 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len $seq_len \
#  --pred_len 192 \
#  --enc_in 8 \
#  --des 'Exp' \
#  --e_layers $e_layers \
#  --d_layers $d_layers \
#  --d_ff $d_ff \
#  --S $S \
#  --itr 1 --batch_size 8 --learning_rate 0.001 >logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'192_el$e_layers'_'dl$d_layers'_'d_ff$d_ff'_'S$S.log
#
#python -u run_longExp.py \
#  --is_training 1 \
#  --root_path ./dataset/exchange_rate/ \
#  --data_path exchange_rate.csv \
#  --model_id Exchange_$seq_len'_'336 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len $seq_len \
#  --pred_len 336 \
#  --enc_in 8 \
#  --des 'Exp' \
#  --e_layers $e_layers \
#  --d_layers $d_layers \
#  --d_ff $d_ff \
#  --S $S \
#  --itr 1 --batch_size 32  --learning_rate 0.001 >logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'336_el$e_layers'_'dl$d_layers'_'d_ff$d_ff'_'S$S.log
#
#python -u run_longExp.py \
#  --is_training 1 \
#  --root_path ./dataset/exchange_rate/ \
#  --data_path exchange_rate.csv \
#  --model_id Exchange_$seq_len'_'720 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len $seq_len \
#  --pred_len 720 \
#  --enc_in 8 \
#  --des 'Exp' \
#  --e_layers $e_layers \
#  --d_layers $d_layers \
#  --d_ff $d_ff \
#  --S $S \
#  --itr 1 --batch_size 32 --learning_rate 0.001 >logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'720_el$e_layers'_'dl$d_layers'_'d_ff$d_ff'_'S$S.log
