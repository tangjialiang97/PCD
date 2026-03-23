export CUDA_VISIBLE_DEVICES=0

# model_name=iTransformer
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
#   --pred_len $pred_len \
#   --label_len 0 \
#   --batch_size 256 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 4 \
#   --dec_in 4 \
#   --c_out 4 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --itr 1
# done
# done

# model_name=PatchTST
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
#   --pred_len $pred_len \
#   --label_len 0 \
#   --batch_size 256 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --n_heads 2 \
#   --enc_in 4 \
#   --dec_in 4 \
#   --c_out 4 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --itr 1
# done
# done

# model_name=TimesNet
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
#   --pred_len $pred_len \
#   --label_len 0 \
#   --batch_size 256 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 4 \
#   --dec_in 4 \
#   --c_out 4 \
#   --d_model 16 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --top_k 5 \
#   --itr 1
# done
# done

model_name=MICN
debug_mode=False
if [ "$debug_mode" = "True" ]; then
  pred_lens=(12)
  data_paths=("flowedb2.csv")
else
  pred_lens=(12 24 48 168)
  data_paths=("flowedb2.csv" "flowedb3.csv" "flowedb4.csv" "flowedb5.csv")
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

# model_name=TimeMixer
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
#   --c_out 4 \
#   --des 'Exp' \
#   --d_model 16 \
#   --d_ff 32 \
#   --down_sampling_layers 3 \
#   --down_sampling_method avg \
#   --itr 1
# done
# done

# model_name=DLinear
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
#   --c_out 4 \
#   --des 'Exp' \
#   --d_model 16 \
#   --d_ff 32 \
#   --d_layers 1 \
#   --itr 1
# done
# done

