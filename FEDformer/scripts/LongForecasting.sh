cd ../../
if [ ! -d "logs" ]; then
  mkdir logs
fi

if [ ! -d "logs/LongForecasting_formers" ]; then
  mkdir logs/LongForecasting_formers
fi
input_len=336
label_len=$[$input_len/2]
# shellcheck disable=SC2164
cd FEDformer

for preLen in 96 192 336 720; do
  # traffic
  python -u run.py \
    --is_training 1 \
    --data_path traffic/traffic.csv \
    --task_id traffic \
    --model FEDformer \
    --data custom \
    --features S \
    --seq_len $input_len \
    --label_len $label_len \
    --pred_len $preLen \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 3 >../logs/LongForecasting_formers/'FEDformer_traffic_'$input_len'_'$preLen.log

  # weather
  python -u run.py \
    --is_training 1 \
    --data_path weather/weather.csv \
    --task_id weather \
    --model FEDformer \
    --data custom \
    --features S \
    --seq_len $input_len \
    --label_len $label_len \
    --pred_len $preLen \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1 >../logs/LongForecasting_formers/'FEDformer_weather_'$input_len'_'$preLen.log
done
