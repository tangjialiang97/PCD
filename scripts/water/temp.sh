export CUDA_VISIBLE_DEVICES=1

model_name=MICN
debug_mode=False
if [ "$debug_mode" = "True" ]; then
  pred_lens=(12)
  data_paths=("flowedb2.csv")
else
  pred_lens=(12 24 48 168)
  data_paths=("flowedb4.csv" "flowedb5.csv")
fi

for data_path in "${data_paths[@]}"
do
for pred_len in "${pred_lens[@]}"
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --inverse \
  --debug $debug_mode \
  --root_path /opt/data/private/data/kdddata/water/flowdb \
  --data_path $data_path \
  --model_id flowdb_256_${pred_len} \
  --model $model_name \
  --data Water \
  --features M \
  --seq_len 256 \
  --pred_len $pred_len \
  --label_len 0 \
  --batch_size 256 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --des 'Exp' \
  --itr 1
done
done

# model_name=TimeFilter
# debug_mode=False
# if [ "$debug_mode" = "True" ]; then
#   pred_lens=(12)
#   data_paths=("flowedb2.csv")
# else
#   pred_lens=(12 24 48 168)
#   data_paths=("flowedb2.csv" "flowedb3.csv" "flowedb4.csv" "flowedb5.csv")
# fi

# for data_path in "${data_paths[@]}"
# do
# for pred_len in "${pred_lens[@]}"
# do

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --inverse \
#   --debug $debug_mode \
#   --root_path /opt/data/private/data/kdddata/water/flowdb \
#   --data_path $data_path \
#   --model_id flowdb_256_${pred_len} \
#   --model $model_name \
#   --data Water \
#   --features M \
#   --seq_len 256 \
#   --batch_size 256 \
#   --pred_len $pred_len \
#   --label_len 0 \
#   --e_layers 2 \
#   --d_model 128 \
#   --d_ff 256 \
#   --n_heads 4 \
#   --patch_len 16 \
#   --alpha 0.1 \
#   --top_p 0.5 \
#   --pos 1 \
#   --dropout 0.1 \
#   --enc_in 4 \
#   --c_out 4 \
#   --des 'Exp' \
#   --itr 1
# done
# done

# model_name=MultiPatchFormer
# debug_mode=False
# if [ "$debug_mode" = "True" ]; then
#   pred_lens=(12)
#   data_paths=("flowedb2.csv")
# else
#   pred_lens=(12 24 48 168)
#   data_paths=("flowedb2.csv" "flowedb3.csv" "flowedb4.csv" "flowedb5.csv")
# fi

# for data_path in "${data_paths[@]}"
# do
# for pred_len in "${pred_lens[@]}"
# do

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --inverse \
#   --debug $debug_mode \
#   --root_path /opt/data/private/data/kdddata/water/flowdb \
#   --data_path $data_path \
#   --model_id flowdb_256_${pred_len} \
#   --model $model_name \
#   --data Water \
#   --features M \
#   --seq_len 256 \
#   --batch_size 256 \
#   --pred_len $pred_len \
#   --label_len 0 \
#   --e_layers 2 \
#   --enc_in 4 \
#   --dec_in 4 \
#   --c_out 4 \
#   --d_model 128 \
#   --d_ff 256 \
#   --des 'Exp' \
#   --itr 1
# done
# done

# model_name=SegRNN
# debug_mode=False
# if [ "$debug_mode" = "True" ]; then
#   pred_lens=(12)
#   data_paths=("flowedb2.csv")
# else
#   pred_lens=(12 24 48 168)
#   data_paths=("flowedb2.csv" "flowedb3.csv" "flowedb4.csv" "flowedb5.csv")
# fi

# for data_path in "${data_paths[@]}"
# do
# for pred_len in "${pred_lens[@]}"
# do

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --inverse \
#   --debug $debug_mode \
#   --root_path /opt/data/private/data/kdddata/water/flowdb \
#   --data_path $data_path \
#   --model_id flowdb_256_${pred_len} \
#   --model $model_name \
#   --data Water \
#   --features M \
#   --seq_len 256 \
#   --seg_len 4 \
#   --batch_size 256 \
#   --pred_len $pred_len \
#   --label_len 0 \
#   --e_layers 2 \
#   --enc_in 4 \
#   --c_out 4 \
#   --des 'Exp' \
#   --itr 1
# done
# done

