cd ../../
# ALL scripts in this file come from Autoformer
if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting_ablations" ]; then
  mkdir ./logs/LongForecasting_ablations
fi

#for model_name in Autoformer Informer Transformer; do
# shellcheck disable=SC1068
model_name=LEAformer_Sparse
# shellcheck disable=SC2034
input_len=96
for pred_len in 96 192 336 720; do
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id traffic'_'$input_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $input_len \
    --pred_len $pred_len \
    --e_layers 4 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --itr 1 \
    --d_ff 2048 \
    --S 512 \
    --d_model 512 \
    --itr 1 \
    --batch_size 16 \
    --learning_rate 0.005 >logs/LongForecasting_ablations/$model_name'_traffic_'$input_len'_'$pred_len.log
done
