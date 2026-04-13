export CUDA_VISIBLE_DEVICES=1

models=(
  DLinear
  iTransformer
)

for model_name in "${models[@]}"
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/water \
  --data_path ResOpsUS_1051.csv \
  --model_id resops_1051_24 \
  --model "$model_name" \
  --data Water \
  --features M \
  --seq_len 256 \
  --pred_len 24 \
  --label_len 0 \
  --batch_size 256 \
  --train_epochs 20 \
  --patience 3 \
  --learning_rate 0.001 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --dropout 0.1 \
  --des 'WaterConservation' \
  --itr 1 \
  --conservation_hidden_dim 128 \
  --conservation_layers 2 \
  --conservation_kernel_size 3 \
  --conservation_delta_t 1.0
done
