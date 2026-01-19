export CUDA_VISIBLE_DEVICES=1

# model_name=iTransformer
# debug_mode=False
# if [ "$debug_mode" = "True" ]; then
#   pred_lens=(12)
#   data_paths=("13PV1.csv")
# else
#   pred_lens=(12 24 48 168)
#   data_paths=("13PV1.csv" "13PV2.csv" "13PV3.csv" "13PV4.csv")
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
#   --root_path /opt/data/private/data/datadbl/solar/13PV/ \
#   --data_path $data_path \
#   --model_id 13PV_256_${pred_len} \
#   --model $model_name \
#   --data Solar \
#   --features S \
#   --seq_len 256 \
#   --pred_len $pred_len \
#   --label_len 0 \
#   --batch_size 256 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --itr 1
# done
# done

model_name=PatchTST
debug_mode=False
if [ m"$debug_mode" = "True" ]; then
  pred_lens=(12)
  data_paths=("13PV1.csv")
else
  pred_lens=(12 24 48 168)
  data_paths=("13PV1.csv" "13PV2.csv" "13PV3.csv" "13PV4.csv")
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
  --root_path /opt/data/private/data/datadbl/solar/13PV/ \
  --data_path $data_path \
  --model_id 13PV_256_${pred_len} \
  --model $model_name \
  --data Solar \
  --features S \
  --seq_len 256 \
  --pred_len $pred_len \
  --label_len 0 \
  --batch_size 256 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --n_heads 2 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1
done
done

model_name=TimesNet
debug_mode=False
if [ "$debug_mode" = "True" ]; then
  pred_lens=(12)
  data_paths=("13PV1.csv")
else
  pred_lens=(12 24 48 168)
  data_paths=("13PV1.csv" "13PV2.csv" "13PV3.csv" "13PV4.csv")
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
  --root_path /opt/data/private/data/datadbl/solar/13PV/ \
  --data_path $data_path \
  --model_id 13PV_256_${pred_len} \
  --model $model_name \
  --data Solar \
  --features S \
  --seq_len 256 \
  --pred_len $pred_len \
  --label_len 0 \
  --batch_size 256 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --top_k 5 \
  --itr 1
done
done

model_name=TimeXer
debug_mode=False
if [ "$debug_mode" = "True" ]; then
  pred_lens=(12)
  data_paths=("13PV1.csv")
else
  pred_lens=(12 24 48 168)
  data_paths=("13PV1.csv" "13PV2.csv" "13PV3.csv" "13PV4.csv")
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
  --root_path /opt/data/private/data/datadbl/solar/13PV/ \
  --data_path $data_path \
  --model_id 13PV_256_${pred_len} \
  --model $model_name \
  --data Solar \
  --features S \
  --seq_len 256 \
  --pred_len $pred_len \
  --label_len 0 \
  --batch_size 256 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1
done
done

model_name=TimeMixer
debug_mode=False
if [ "$debug_mode" = "True" ]; then
  pred_lens=(12)
  data_paths=("13PV1.csv")
else
  pred_lens=(12 24 48 168)
  data_paths=("13PV1.csv" "13PV2.csv" "13PV3.csv" "13PV4.csv")
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
  --root_path /opt/data/private/data/datadbl/solar/13PV/ \
  --data_path $data_path \
  --model_id 13PV_256_${pred_len} \
  --model $model_name \
  --data Solar \
  --features S \
  --seq_len 256 \
  --batch_size 256 \
  --pred_len $pred_len \
  --label_len 0 \
  --e_layers 2 \
  --enc_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 16 \
  --d_ff 32 \
  --down_sampling_layers 3 \
  --down_sampling_method avg \
  --itr 1
done
done

model_name=DLinear
debug_mode=False
if [ "$debug_mode" = "True" ]; then
  pred_lens=(12)
  data_paths=("13PV1.csv")
else
  pred_lens=(12 24 48 168)
  data_paths=("13PV1.csv" "13PV2.csv" "13PV3.csv" "13PV4.csv")
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
  --root_path /opt/data/private/data/datadbl/solar/13PV/ \
  --data_path $data_path \
  --model_id 13PV_256_${pred_len} \
  --model $model_name \
  --data Solar \
  --features S \
  --seq_len 256 \
  --batch_size 256 \
  --pred_len $pred_len \
  --label_len 0 \
  --e_layers 2 \
  --enc_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 16 \
  --d_ff 32 \
  --d_layers 1 \
  --itr 1
done
done

model_name=SegRNN
debug_mode=False
if [ "$debug_mode" = "True" ]; then
  pred_lens=(12)
  data_paths=("13PV1.csv")
else
  pred_lens=(12 24 48 168)
  data_paths=("13PV1.csv" "13PV2.csv" "13PV3.csv" "13PV4.csv")
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
  --root_path /opt/data/private/data/datadbl/solar/13PV/ \
  --data_path $data_path \
  --model_id 13PV_256_${pred_len} \
  --model $model_name \
  --data Solar \
  --features S \
  --seq_len 256 \
  --seg_len 4 \
  --batch_size 256 \
  --pred_len $pred_len \
  --label_len 0 \
  --e_layers 2 \
  --enc_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1
done
done

model_name=TimeFilter
debug_mode=False
if [ "$debug_mode" = "True" ]; then
  pred_lens=(12)
  data_paths=("13PV1.csv")
else
  pred_lens=(12 24 48 168)
  data_paths=("13PV1.csv" "13PV2.csv" "13PV3.csv" "13PV4.csv")
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
  --root_path /opt/data/private/data/datadbl/solar/13PV/ \
  --data_path $data_path \
  --model_id 13PV_256_${pred_len} \
  --model $model_name \
  --data Solar \
  --features S \
  --seq_len 256 \
  --batch_size 256 \
  --pred_len $pred_len \
  --label_len 0 \
  --e_layers 2 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 4 \
  --patch_len 16 \
  --alpha 0.1 \
  --top_p 0.5 \
  --pos 1 \
  --dropout 0.1 \
  --enc_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1
done
done

model_name=MultiPatchFormer
debug_mode=False
if [ "$debug_mode" = "True" ]; then
  pred_lens=(12)
  data_paths=("13PV1.csv")
else
  pred_lens=(12 24 48 168)
  data_paths=("13PV1.csv" "13PV2.csv" "13PV3.csv" "13PV4.csv")
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
  --root_path /opt/data/private/data/datadbl/solar/13PV/ \
  --data_path $data_path \
  --model_id 13PV_256_${pred_len} \
  --model $model_name \
  --data Solar \
  --features S \
  --seq_len 256 \
  --batch_size 256 \
  --pred_len $pred_len \
  --label_len 0 \
  --e_layers 2 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 1
done
done

model_name=MambaSimple
debug_mode=False
if [ "$debug_mode" = "True" ]; then
  pred_lens=(12)
  data_paths=("13PV1.csv")
else
  pred_lens=(12 24 48 168)
  data_paths=("13PV1.csv" "13PV2.csv" "13PV3.csv" "13PV4.csv")
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
  --root_path /opt/data/private/data/datadbl/solar/13PV/ \
  --data_path $data_path \
  --model_id 13PV_256_${pred_len} \
  --model $model_name \
  --data Solar \
  --features S \
  --seq_len 256 \
  --batch_size 256 \
  --pred_len $pred_len \
  --label_len 0 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 1 \
  --expand 2 \
  --d_ff 16 \
  --d_conv 4 \
  --c_out 1 \
  --d_model 128 \
  --des 'Exp' \
  --itr 1
done
done