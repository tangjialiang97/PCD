export CUDA_VISIBLE_DEVICES=0

# ============================================================
# Reservoir Conservation Physics Experiments (ResOpsUS)
# ============================================================
# Three-way comparison:
#   1. DLinear (backbone only)
#   2. DLinear + soft conservation loss (use_auxiliary)
#   3. DLinearReservoirPhysics (PCD hard constraint)
#   4. DLinearReservoirPhysics + auxiliary conservation loss
# ============================================================

debug_mode=False

if [ "$debug_mode" = "True" ]; then
  pred_lens=(12)
  data_paths=("ResOpsUS_1869.csv")
else
  pred_lens=(24)
  data_paths=("ResOpsUS_1051.csv")
fi

# --- 1. Backbone: DLinear ---
model_name=DLinear
for data_path in "${data_paths[@]}"
do
for pred_len in "${pred_lens[@]}"
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --inverse \
  --debug $debug_mode \
  --root_path ./data/water \
  --data_path $data_path \
  --model_id ResOpsUS_256_${pred_len} \
  --model $model_name \
  --data Water \
  --features M \
  --seq_len 256 \
  --pred_len $pred_len \
  --label_len 0 \
  --batch_size 128 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --learning_rate 0.001 \
  --train_epochs 20 \
  --patience 5 \
  --itr 1
done
done

# --- 2. PCD: DLinearReservoirPhysics ---
model_name=DLinearReservoirPhysics
for data_path in "${data_paths[@]}"
do
for pred_len in "${pred_lens[@]}"
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --inverse \
  --debug $debug_mode \
  --root_path ./data/water \
  --data_path $data_path \
  --model_id ResOpsUS_256_${pred_len} \
  --model $model_name \
  --data Water \
  --features M \
  --seq_len 256 \
  --pred_len $pred_len \
  --label_len 0 \
  --batch_size 128 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --learning_rate 0.001 \
  --train_epochs 20 \
  --patience 5 \
  --physics_hidden_dim 128 \
  --physics_layers 2 \
  --physics_kernel_size 3 \
  --reservoir_use_slack \
  --reservoir_delta_t 1.0 \
  --itr 1
done
done

# --- 3. PCD + Auxiliary: DLinearReservoirPhysics + conservation loss ---
model_name=DLinearReservoirPhysics
for data_path in "${data_paths[@]}"
do
for pred_len in "${pred_lens[@]}"
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --inverse \
  --debug $debug_mode \
  --root_path ./data/water \
  --data_path $data_path \
  --model_id ResOpsUS_256_${pred_len}_aux \
  --model $model_name \
  --data Water \
  --features M \
  --seq_len 256 \
  --pred_len $pred_len \
  --label_len 0 \
  --batch_size 128 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --learning_rate 0.001 \
  --train_epochs 20 \
  --patience 5 \
  --physics_hidden_dim 128 \
  --physics_layers 2 \
  --physics_kernel_size 3 \
  --reservoir_use_slack \
  --reservoir_delta_t 1.0 \
  --use_auxiliary \
  --conservation_weight 1.0 \
  --reservoir_warmup_epochs 2 \
  --itr 1
done
done
