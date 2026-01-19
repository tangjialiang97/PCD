export CUDA_VISIBLE_DEVICES=1


model_name=Sundial

seq_len=512

for pred_len in 8
do
python -u run.py \
  --task_name zero_shot_forecast \
  --is_training 0 \
  --root_path /opt/data/private/data/timedata/datasets/Stanford/ \
  --data_path Stanford.csv \
  --model_id Stanford_$seq_len'_'$pred_len \
  --model $model_name \
  --data Stanford \
  --features S \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 1 \
  --d_model 512 \
  --dropout 0.5 \
  --learning_rate 0.0001 \
  --des 'Exp' \
  --itr 1
done