if [ ! -d "logs" ]; then
    mkdir ./logs
fi

if [ ! -d "logs/LongForecasting" ]; then
    mkdir logs/LongForecasting
fi
seq_len=36
model_name=MEAformer
e_layers=3
d_layers=1
d_ff=512
S=512
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id national_illness_$seq_len'_'24 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 18 \
  --pred_len 24 \
  --enc_in 7 \
  --des 'Exp' \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --d_ff $d_ff \
  --S $S \
  --itr 1 --batch_size 32 --learning_rate 0.01 >logs/LongForecasting/$model_name'_'ili_$seq_len'_'24_el$e_layers'_'dl$d_layers'_'d_ff$d_ff'_'S$S.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id national_illness_$seq_len'_'36 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 18 \
  --pred_len 36 \
  --enc_in 7 \
  --des 'Exp' \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --d_ff $d_ff \
  --S $S \
  --itr 1 --batch_size 32 --learning_rate 0.01  >logs/LongForecasting/$model_name'_'ili_$seq_len'_'36_el$e_layers'_'dl$d_layers'_'d_ff$d_ff'_'S$S.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id national_illness_$seq_len'_'48 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 18 \
  --pred_len 48 \
  --enc_in 7 \
  --des 'Exp' \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --d_ff $d_ff \
  --S $S \
  --itr 1 --batch_size 32 --learning_rate 0.01  >logs/LongForecasting/$model_name'_'ili_$seq_len'_'48_el$e_layers'_'dl$d_layers'_'d_ff$d_ff'_'S$S.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id national_illness_$seq_len'_'60 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 18 \
  --pred_len 60 \
  --enc_in 7 \
  --des 'Exp' \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --d_ff $d_ff \
  --S $S \
  --itr 1 --batch_size 32 --learning_rate 0.01  >logs/LongForecasting/$model_name'_'ili_$seq_len'_'60_el$e_layers'_'dl$d_layers'_'d_ff$d_ff'_'S$S.log
