
if [ ! -d "logs" ]; then
    mkdir ./logs
fi

if [ ! -d "logs/LongForecasting" ]; then
    mkdir logs/LongForecasting
fi
seq_len=96
model_name=MEAformer
e_layers=5
d_layers=1
d_ff=2048
S=512
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 321 \
  --des 'Exp' \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --d_ff $d_ff \
  --S $S \
  --itr 1 --batch_size 16  --learning_rate 0.001 >logs/LongForecasting/$model_name'_'electricity_$seq_len'_'96_el$e_layers'_'dl$d_layers'_'d_ff$d_ff'_'S$S.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_'192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 321 \
  --des 'Exp' \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --d_ff $d_ff \
  --S $S \
  --itr 1 --batch_size 16  --learning_rate 0.001 >logs/LongForecasting/$model_name'_'electricity_$seq_len'_'192_el$e_layers'_'dl$d_layers'_'d_ff$d_ff'_'S$S.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_'336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 321 \
  --des 'Exp' \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --d_ff $d_ff \
  --S $S \
  --itr 1 --batch_size 16  --learning_rate 0.001  >logs/LongForecasting/$model_name'_'electricity_$seq_len'_'336_el$e_layers'_'dl$d_layers'_'d_ff$d_ff'_'S$S.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_'720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 321 \
  --des 'Exp' \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --d_ff $d_ff \
  --S $S \
  --itr 1 --batch_size 16  --learning_rate 0.001  >logs/LongForecasting/$model_name'_'electricity_$seq_len'_'720_el$e_layers'_'dl$d_layers'_'d_ff$d_ff'_'S$S.log
