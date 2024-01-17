
if [ ! -d "logs" ]; then
    mkdir ./logs
fi

if [ ! -d "logs/LongForecasting" ]; then
    mkdir logs/LongForecasting
fi
seq_len=96
model_name=MEAformer
e_layers=2
d_layers=1
d_ff=512
S=64
#24 48 168 336 720
for preLen in 24 48 168 336 720; do

  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_$seq_len'_'$preLen \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --pred_len $preLen \
    --enc_in 7 \
    --des 'Exp' \
    --e_layers $e_layers \
    --d_layers $d_layers \
    --d_ff $d_ff \
    --S $S \
    --itr 1 --batch_size 32 --learning_rate 0.001 >logs/LongForecasting/$model_name'_'Etth1_$seq_len'_'$preLen'_'el$e_layers'_'dl$d_layers'_'d_ff$d_ff'_'S$S.log
done
