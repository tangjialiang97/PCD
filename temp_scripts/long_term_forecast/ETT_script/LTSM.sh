export CUDA_VISIBLE_DEVICES=0

model_name=Chronos2


for seq_len in 128 256 512
do
for pred_len in 8 16 24
do
python -u run.py \
  --task_name zero_shot_forecast \
  --is_training 0 \
  --root_path /opt/data/private/data/Folsom/ \
  --data_path_trainval Folsom_irradiance_train.csv \
  --data_path_test Folsom_irradiance_test.csv \
  --model_id Stanford_$seq_len'_'$pred_len \
  --model $model_name \
  --data Irr \
  --features S \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 1 \
  --d_model 512 \
  --dropout 0.5 \
  --inverse \
  --learning_rate 0.0001 \
  --des 'Exp' \
  --itr 1
done
done

# model_name=TiRex


# for seq_len in 128 256 512
# do
# for pred_len in 8 16 24
# do
# python -u run.py \
#   --task_name zero_shot_forecast \
#   --is_training 0 \
#   --root_path /opt/data/private/data/timedata/datasets/Stanford/ \
#   --data_path Stanford.csv \
#   --model_id Stanford_$seq_len'_'$pred_len \
#   --model $model_name \
#   --data Stanford \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len $pred_len \
#   --label_len 0 \
#   --enc_in 1 \
#   --d_model 512 \
#   --dropout 0.5 \
#   --inverse \
#   --learning_rate 0.0001 \
#   --des 'Exp' \
#   --itr 1
# done
# done
# seq_len=4

# for pred_len_hour in 2
# do
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 0 \
#   --root_path /opt/data/private/data/Folsom/ \
#   --data_path Target_intra-hour_filtered.csv \
#   --model_id FOLSOM_$seq_len'_'$pred_len_hour \
#   --model $model_name \
#   --data Folsom \
#   --features M \
#   --seq_len_hours $seq_len \
#   --pred_len_hours $pred_len_hour \
#   --label_len_hours 0 \
#   --enc_in 42 \
#   --d_model 512 \
#   --dropout 0.5 \
#   --learning_rate 0.0001 \
#   --des 'Exp' \
#   --itr 1
# done
# for pred_len in 96 192 336 720
# do
# python -u run.py \
#   --task_name zero_shot_forecast \
#   --is_training 0 \
#   --root_path /opt/data/private/data/timedata/datasets/ETT-small \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_$seq_len'_'$pred_len \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len $pred_len \
#   --seg_len 24 \
#   --enc_in 7 \
#   --d_model 512 \
#   --dropout 0.5 \
#   --learning_rate 0.0001 \
#   --des 'Exp' \
#   --itr 1
# done

# for pred_len in 96 192 336 720
# do
# python -u run.py \
#   --task_name zero_shot_forecast \
#   --is_training 0 \
#   --root_path /opt/data/private/data/timedata/datasets/ETT-small \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_$seq_len'_'$pred_len \
#   --model $model_name \
#   --data ETTh2 \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len $pred_len \
#   --seg_len 24 \
#   --enc_in 7 \
#   --d_model 256 \
#   --dropout 0.5 \
#   --learning_rate 0.0001 \
#   --des 'Exp' \
#   --itr 1
# done


# for pred_len in 192 336 720
# do
# python -u run.py \
#   --task_name zero_shot_forecast \
#   --is_training 0 \
#   --root_path /opt/data/private/data/timedata/datasets/ETT-small \
#   --data_path ETTm1.csv \
#   --model_id ETTm1_$seq_len'_'$pred_len \
#   --model $model_name \
#   --data ETTm1 \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len $pred_len \
#   --seg_len 24 \
#   --enc_in 7 \
#   --d_model 512 \
#   --dropout 0.5 \
#   --learning_rate 0.0001 \
#   --des 'Exp' \
#   --itr 1
# done


# for pred_len in 96 192 336 720
# do
# python -u run.py \
#   --task_name zero_shot_forecast \
#   --is_training 0 \
#   --root_path /opt/data/private/data/timedata/datasets/ETT-small \
#   --data_path ETTm2.csv \
#   --model_id ETTm2_$seq_len'_'$pred_len \
#   --model $model_name \
#   --data ETTm2 \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len $pred_len \
#   --seg_len 24 \
#   --enc_in 7 \
#   --d_model 512 \
#   --dropout 0.5 \
#   --learning_rate 0.0001 \
#   --des 'Exp' \
#   --itr 1
# done

# for pred_len in 96 192 336 720
# do
# python -u run.py \
#   --task_name zero_shot_forecast \
#   --is_training 0 \
#   --root_path /opt/data/private/data/timedata/datasets/electricity \
#   --data_path electricity.csv \
#   --model_id electricity_$seq_len'_'$pred_len \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len $pred_len \
#   --seg_len 24 \
#   --enc_in 321 \
#   --d_model 512 \
#   --dropout 0.5 \
#   --learning_rate 0.0001 \
#   --des 'Exp' \
#   --itr 1
# done

# for pred_len in 96 192 336 720
# do
# python -u run.py \
#   --task_name zero_shot_forecast \
#   --is_training 0 \
#   --root_path /opt/data/private/data/timedata/datasets/exchange_rate \
#   --data_path exchange_rate.csv \
#   --model_id exchange_rate_$seq_len'_'$pred_len \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len $pred_len \
#   --seg_len 24 \
#   --enc_in 8 \
#   --d_model 512 \
#   --dropout 0.5 \
#   --learning_rate 0.0001 \
#   --des 'Exp' \
#   --itr 1
# done

# for pred_len in 96 192 336 720
# do
# python -u run.py \
#   --task_name zero_shot_forecast \
#   --is_training 0 \
#   --root_path /opt/data/private/data/timedata/datasets/traffic \
#   --data_path traffic.csv \
#   --model_id traffic_$seq_len'_'$pred_len \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len $pred_len \
#   --seg_len 24 \
#   --enc_in 862 \
#   --d_model 512 \
#   --dropout 0.5 \
#   --learning_rate 0.0001 \
#   --des 'Exp' \
#   --itr 1
# done

# for pred_len in 96 192 336 720
# do
# python -u run.py \
#   --task_name zero_shot_forecast \
#   --is_training 0 \
#   --root_path /opt/data/private/data/timedata/datasets/weather \
#   --data_path weather.csv \
#   --model_id weather_$seq_len'_'$pred_len \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len $pred_len \
#   --seg_len 24 \
#   --enc_in 21 \
#   --d_model 512 \
#   --dropout 0.5 \
#   --learning_rate 0.0001 \
#   --des 'Exp' \
#   --itr 1
# done