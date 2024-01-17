cd ../../
#cd ..

input_len=96
label_len=$(($input_len / 2))
model=FEDformer
cd FEDformer-master
preLen=96
#for preLen in 24 48 168; do

  # ETTh1
#  python -u run.py \
#    --is_training 1 \
#    --root_path ../dataset/ETT-small/ \
#    --data_path ETTh1.csv \
#    --task_id ETTh1 \
#    --model $model \
#    --data ETTh1 \
#    --features M \
#    --seq_len 96 \
#    --label_len 48 \
#    --pred_len $preLen \
#    --e_layers 2 \
#    --d_layers 1 \
#    --factor 3 \
#    --enc_in 7 \
#    --dec_in 7 \
#    --c_out 7 \
#    --des 'Exp' \
#    --d_model 512 \
#    --itr 1  >../logs/LongForecasting_formers/$model'_'Etth1_96'_'$preLen.log
#
#
#
#  # ETTh2
#  python -u run.py \
#    --is_training 1 \
#    --root_path ../dataset/ETT-small/ \
#    --data_path ETTh2.csv \
#    --task_id ETTh2 \
#    --model $model \
#    --data ETTh2 \
#    --features M \
#    --seq_len 96 \
#    --label_len 48 \
#    --pred_len $preLen \
#    --e_layers 2 \
#    --d_layers 1 \
#    --factor 3 \
#    --enc_in 7 \
#    --dec_in 7 \
#    --c_out 7 \
#    --des 'Exp' \
#    --d_model 512 \
#    --itr 1  >../logs/LongForecasting_formers/$model'_'Etth2_96'_'$preLen.log
#done


for preLen in 288 672; do
  # ETT m1
  python -u run.py \
    --is_training 1 \
    --root_path ../dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --task_id ETTm1 \
    --model $model \
    --data ETTm1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $preLen \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 512 \
    --itr 1  >../logs/LongForecasting_formers/$model'_'Ettm1_96'_'$preLen.log
  # ETTm2
  python -u run.py \
    --is_training 1 \
    --root_path ../dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --task_id ETTm2 \
    --model $model \
    --data ETTm2 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $preLen \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 512 \
    --itr 1  >../logs/LongForecasting_formers/$model'_'Ettm2_96'_'$preLen.log
done