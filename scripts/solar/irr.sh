<<<<<<< HEAD

# model_name=iTransformer
# debug_mode=False
# if [ "$debug_mode" = "True" ]; then
#   pred_lens=(12)
#   data_paths=("folsom.csv")
# else
#   pred_lens=(24)
#   data_paths=("Folsom_combined_no_zero.csv")
# fi

# for data_path in "${data_paths[@]}"
# do
# for pred_len in "${pred_lens[@]}"
# do

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --use_auxiliary \
#   --debug $debug_mode \
#   --root_path /root/tjl/data/newdata/ \
#   --data_path $data_path \
#   --model_id irr_256_${pred_len} \
#   --model $model_name \
#   --data Solar \
#   --features M \
#   --seq_len 256 \
#   --pred_len $pred_len \
#   --label_len 0 \
#   --batch_size 512 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 3 \
#   --dec_in 3 \
#   --c_out 3 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --itr 1 \
#   --gpu 4
# done
# done


# model_name=PatchTST
# debug_mode=False
# if [ m"$debug_mode" = "True" ]; then
#   pred_lens=(12)
#   data_paths=("folsom.csv")
# else
#   pred_lens=(12 24 48 168)
#     data_paths=("Folsom_combined_no_zero.csv")
# fi

# for data_path in "${data_paths[@]}"
# do
# for pred_len in "${pred_lens[@]}"
# do

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --use_auxiliary \
#   --debug $debug_mode \
#   --root_path /root/tjl/data/newdata/ \
#   --data_path $data_path \
#   --model_id irr_256_${pred_len} \
#   --model $model_name \
#   --data Solar \
#   --features S \
#   --seq_len 256 \
#   --pred_len $pred_len \
#   --label_len 0 \
#   --batch_size 512 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --n_heads 2 \
#   --enc_in 3 \
#   --dec_in 3 \
#   --c_out 3 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --itr 1 \
#   --gpu 4
# done
# done

# model_name=TimesNet
# debug_mode=False
# if [ "$debug_mode" = "True" ]; then
#   pred_lens=(12)
#   data_paths=("folsom.csv")
# else
#   pred_lens=(12 24 48 168)
#     data_paths=("Folsom_combined_no_zero.csv")
# fi

# for data_path in "${data_paths[@]}"
# do
# for pred_len in "${pred_lens[@]}"
# do

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --use_auxiliary \
#   --debug $debug_mode \
#   --root_path /root/tjl/data/newdata/ \
#   --data_path $data_path \
#   --model_id irr_256_${pred_len} \
#   --model $model_name \
#   --data Solar \
#   --features S \
#   --seq_len 256 \
#   --pred_len $pred_len \
#   --label_len 0 \
#   --batch_size 512 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 3 \
#   --dec_in 3 \
#   --c_out 3 \
#   --d_model 16 \
#   --d_ff 32 \
#   --des 'Exp' \
#   --top_k 5 \
#   --itr 1 \
#   --gpu 4
# done
# done

# model_name=TimeXer
# debug_mode=False
# if [ "$debug_mode" = "True" ]; then
#   pred_lens=(12)
#   data_paths=("folsom.csv")
# else
#   pred_lens=(12 24 48 168)
#     data_paths=("Folsom_combined_no_zero.csv")
# fi

# for data_path in "${data_paths[@]}"
# do
# for pred_len in "${pred_lens[@]}"
# do

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --use_auxiliary \
#   --debug $debug_mode \
#   --root_path /root/tjl/data/newdata/ \
#   --data_path $data_path \
#   --model_id irr_256_${pred_len} \
#   --model $model_name \
#   --data Solar \
#   --features S \
#   --seq_len 256 \
#   --pred_len $pred_len \
#   --label_len 0 \
#   --batch_size 512 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 3 \
#   --dec_in 3 \
#   --c_out 3 \
#   --des 'Exp' \
#   --itr 1 \
#   --gpu 4
# done
# done

# model_name=TimeMixer
# debug_mode=False
# if [ "$debug_mode" = "True" ]; then
#   pred_lens=(12)
#   data_paths=("folsom.csv")
# else
#   pred_lens=(12 24 48 168)
#     data_paths=("Folsom_combined_no_zero.csv")
# fi

# for data_path in "${data_paths[@]}"
# do
# for pred_len in "${pred_lens[@]}"
# do

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --use_auxiliary \
#   --debug $debug_mode \
#   --root_path /root/tjl/data/newdata/ \
#   --data_path $data_path \
#   --model_id irr_256_${pred_len} \
#   --model $model_name \
#   --data Solar \
#   --features S \
#   --seq_len 256 \
#   --batch_size 512 \
#   --pred_len $pred_len \
#   --label_len 0 \
#   --e_layers 2 \
#   --enc_in 3 \
#   --c_out 3 \
#   --des 'Exp' \
#   --d_model 16 \
#   --d_ff 32 \
#   --down_sampling_layers 3 \
#   --down_sampling_method avg \
#   --itr 1 \
#   --gpu 4
# done
# done

# model_name=DLinear
# debug_mode=False
# if [ "$debug_mode" = "True" ]; then
#   pred_lens=(12)
#   data_paths=("folsom.csv")
# else
#   pred_lens=(12 24 48 168)
#     data_paths=("Folsom_combined_no_zero.csv")
# fi

# for data_path in "${data_paths[@]}"
# do
# for pred_len in "${pred_lens[@]}"
# do

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --use_auxiliary \
#   --debug $debug_mode \
#   --root_path /root/tjl/data/newdata/ \
#   --data_path $data_path \
#   --model_id irr_256_${pred_len} \
#   --model $model_name \
#   --data Solar \
#   --features S \
#   --seq_len 256 \
#   --batch_size 512 \
#   --pred_len $pred_len \
#   --label_len 0 \
#   --e_layers 2 \
#   --enc_in 3 \
#   --c_out 3 \
#   --des 'Exp' \
#   --d_model 16 \
#   --d_ff 32 \
#   --d_layers 1 \
#   --itr 1 \
#   --gpu 4
# done
# done

# model_name=SegRNN
# debug_mode=False
# if [ "$debug_mode" = "True" ]; then
#   pred_lens=(12)
#   data_paths=("folsom.csv")
# else
#   pred_lens=(12 24 48 168)
#     data_paths=("Folsom_combined_no_zero.csv")
# fi

# for data_path in "${data_paths[@]}"
# do
# for pred_len in "${pred_lens[@]}"
# do

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --use_auxiliary \
#   --debug $debug_mode \
#   --root_path /root/tjl/data/newdata/ \
#   --data_path $data_path \
#   --model_id irr_256_${pred_len} \
#   --model $model_name \
#   --data Solar \
#   --features S \
#   --seq_len 256 \
#   --seg_len 4 \
#   --batch_size 512 \
#   --pred_len $pred_len \
#   --label_len 0 \
#   --e_layers 2 \
#   --enc_in 3 \
#   --c_out 3 \
#   --des 'Exp' \
#   --itr 1 \
#   --gpu 4
# done
# done

# model_name=TimeFilter
# debug_mode=False
# if [ "$debug_mode" = "True" ]; then
#   pred_lens=(12)
#   data_paths=("folsom.csv")
# else
#   pred_lens=(12 24 48 168)
#     data_paths=("Folsom_combined_no_zero.csv")
# fi

# for data_path in "${data_paths[@]}"
# do
# for pred_len in "${pred_lens[@]}"
# do

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --use_auxiliary \
#   --debug $debug_mode \
#   --root_path /root/tjl/data/newdata/ \
#   --data_path $data_path \
#   --model_id irr_256_${pred_len} \
#   --model $model_name \
#   --data Solar \
#   --features S \
#   --seq_len 256 \
#   --batch_size 512 \
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
#   --enc_in 3 \
#   --c_out 3 \
#   --des 'Exp' \
#   --itr 1 \
#   --gpu 4
# done
# done

# model_name=MultiPatchFormer
# debug_mode=False
# if [ "$debug_mode" = "True" ]; then
#   pred_lens=(12)
#   data_paths=("folsom.csv")
# else
#   pred_lens=(12 24 48 168)
#     data_paths=("Folsom_combined_no_zero.csv")
# fi

# for data_path in "${data_paths[@]}"
# do
# for pred_len in "${pred_lens[@]}"
# do

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --use_auxiliary \
#   --debug $debug_mode \
#   --root_path /root/tjl/data/newdata/ \
#   --data_path $data_path \
#   --model_id irr_256_${pred_len} \
#   --model $model_name \
#   --data Solar \
#   --features S \
#   --seq_len 256 \
#   --batch_size 512 \
#   --pred_len $pred_len \
#   --label_len 0 \
#   --e_layers 2 \
#   --enc_in 3 \
#   --dec_in 3 \
#   --c_out 3 \
#   --d_model 128 \
#   --d_ff 256 \
#   --des 'Exp' \
#   --itr 1 \
#   --gpu 4
# done
# done

#!/usr/bin/env bash
set -euo pipefail

model_name=iTransformer
debug_mode=False
gpu_id=4
log_file="scripts/solar/irr1_grid.log"

=======
export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer
debug_mode=False
if [ "$debug_mode" = "True" ]; then
  pred_lens=(12)
  data_paths=("folsom.csv")
else
  pred_lens=(12 24 48 168)
  data_paths=("folsom.csv" "stanford.csv")
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
  --root_path /opt/data/private/data/datadbl/solar/irr/ \
  --data_path $data_path \
  --model_id irr_256_${pred_len} \
  --model $model_name \
  --data Solar \
  --features S \
  --seq_len 256 \
  --pred_len $pred_len \
  --label_len 0 \
  --batch_size 512 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1
done
done

model_name=PatchTST
debug_mode=False
if [ m"$debug_mode" = "True" ]; then
  pred_lens=(12)
  data_paths=("folsom.csv")
else
  pred_lens=(12 24 48 168)
  data_paths=("folsom.csv" "stanford.csv")
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
  --root_path /opt/data/private/data/datadbl/solar/irr/ \
  --data_path $data_path \
  --model_id irr_256_${pred_len} \
  --model $model_name \
  --data Solar \
  --features S \
  --seq_len 256 \
  --pred_len $pred_len \
  --label_len 0 \
  --batch_size 512 \
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
  data_paths=("folsom.csv")
else
  pred_lens=(12 24 48 168)
  data_paths=("folsom.csv" "stanford.csv")
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
  --root_path /opt/data/private/data/datadbl/solar/irr/ \
  --data_path $data_path \
  --model_id irr_256_${pred_len} \
  --model $model_name \
  --data Solar \
  --features S \
  --seq_len 256 \
  --pred_len $pred_len \
  --label_len 0 \
  --batch_size 512 \
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
  data_paths=("folsom.csv")
else
  pred_lens=(12 24 48 168)
  data_paths=("folsom.csv" "stanford.csv")
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
  --root_path /opt/data/private/data/datadbl/solar/irr/ \
  --data_path $data_path \
  --model_id irr_256_${pred_len} \
  --model $model_name \
  --data Solar \
  --features S \
  --seq_len 256 \
  --pred_len $pred_len \
  --label_len 0 \
  --batch_size 512 \
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
>>>>>>> 43bc95b0a4ddd08689931ee59d6b5b814a8c9ed1
if [ "$debug_mode" = "True" ]; then
  pred_lens=(12)
  data_paths=("folsom.csv")
else
<<<<<<< HEAD
  pred_lens=(12 48 168)
  data_paths=("Folsom_combined_no_zero.csv")
fi

# Grid search space (log-scaled for stable loss balancing)
closure_weights=(0.0 1.0 0.1 0.01)
limit_weights=(0.0 1.0 0.1 0.01)
var_weights=(0.0 1.0 0.1 0.01)
var_gammas=(0.03 0.1 0.3 1.0)

echo "===== Grid search started: $(date '+%F %T') =====" | tee -a "$log_file"

for data_path in "${data_paths[@]}"; do
for pred_len in "${pred_lens[@]}"; do
for closure_weight in "${closure_weights[@]}"; do
for limit_weight in "${limit_weights[@]}"; do
for var_weight in "${var_weights[@]}"; do
for var_gamma in "${var_gammas[@]}"; do

  echo "[RUN] data=${data_path} pred_len=${pred_len} cw=${closure_weight} lw=${limit_weight} vw=${var_weight} vg=${var_gamma}" | tee -a "$log_file"

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --use_auxiliary \
    --closure_weight "$closure_weight" \
    --limit_weight "$limit_weight" \
    --var_weight "$var_weight" \
    --var_gamma "$var_gamma" \
    --debug "$debug_mode" \
    --root_path /root/tjl/data/newdata/ \
    --data_path "$data_path" \
    --model_id "irr_256_${pred_len}" \
    --model "$model_name" \
    --data Solar \
    --features M \
    --seq_len 256 \
    --pred_len "$pred_len" \
    --label_len 0 \
    --batch_size 512 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 3 \
    --dec_in 3 \
    --c_out 3 \
    --des 'ExpGrid' \
    --d_model 128 \
    --d_ff 128 \
    --itr 1 \
    --gpu "$gpu_id" 2>&1 | tee -a "$log_file"

done
done
done
done
done
done

echo "===== Grid search finished: $(date '+%F %T') =====" | tee -a "$log_file"
=======
  pred_lens=(12 24 48 168)
  data_paths=("folsom.csv" "stanford.csv")
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
  --root_path /opt/data/private/data/datadbl/solar/irr/ \
  --data_path $data_path \
  --model_id irr_256_${pred_len} \
  --model $model_name \
  --data Solar \
  --features S \
  --seq_len 256 \
  --batch_size 512 \
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
  data_paths=("folsom.csv")
else
  pred_lens=(12 24 48 168)
  data_paths=("folsom.csv" "stanford.csv")
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
  --root_path /opt/data/private/data/datadbl/solar/irr/ \
  --data_path $data_path \
  --model_id irr_256_${pred_len} \
  --model $model_name \
  --data Solar \
  --features S \
  --seq_len 256 \
  --batch_size 512 \
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
  data_paths=("folsom.csv")
else
  pred_lens=(12 24 48 168)
  data_paths=("folsom.csv" "stanford.csv")
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
  --root_path /opt/data/private/data/datadbl/solar/irr/ \
  --data_path $data_path \
  --model_id irr_256_${pred_len} \
  --model $model_name \
  --data Solar \
  --features S \
  --seq_len 256 \
  --seg_len 4 \
  --batch_size 512 \
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
  data_paths=("folsom.csv")
else
  pred_lens=(12 24 48 168)
  data_paths=("folsom.csv" "stanford.csv")
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
  --root_path /opt/data/private/data/datadbl/solar/irr/ \
  --data_path $data_path \
  --model_id irr_256_${pred_len} \
  --model $model_name \
  --data Solar \
  --features S \
  --seq_len 256 \
  --batch_size 512 \
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
  data_paths=("folsom.csv")
else
  pred_lens=(12 24 48 168)
  data_paths=("folsom.csv" "stanford.csv")
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
  --root_path /opt/data/private/data/datadbl/solar/irr/ \
  --data_path $data_path \
  --model_id irr_256_${pred_len} \
  --model $model_name \
  --data Solar \
  --features S \
  --seq_len 256 \
  --batch_size 512 \
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
  data_paths=("folsom.csv")
else
  pred_lens=(12 24 48 168)
  data_paths=("folsom.csv" "stanford.csv")
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
  --root_path /opt/data/private/data/datadbl/solar/irr/ \
  --data_path $data_path \
  --model_id irr_256_${pred_len} \
  --model $model_name \
  --data Solar \
  --features S \
  --seq_len 256 \
  --batch_size 512 \
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
>>>>>>> 43bc95b0a4ddd08689931ee59d6b5b814a8c9ed1
