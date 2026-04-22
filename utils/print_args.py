def print_args(args):
    print("\033[1m" + "Basic Config" + "\033[0m")
    print(f'  {"Task Name:":<20}{args.task_name:<20}{"Is Training:":<20}{args.is_training:<20}')
    print(f'  {"Model ID:":<20}{args.model_id:<20}{"Model:":<20}{args.model:<20}')
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    print(f'  {"Data:":<20}{args.data:<20}{"Root Path:":<20}{args.root_path:<20}')
    print(f'  {"Data Path:":<20}{args.data_path:<20}{"Features:":<20}{args.features:<20}')
    print(f'  {"Target:":<20}{args.target:<20}{"Freq:":<20}{args.freq:<20}')
    print(f'  {"Checkpoints:":<20}{args.checkpoints:<20}')
    print()

    if args.task_name in ['long_term_forecast', 'short_term_forecast']:
        print("\033[1m" + "Forecasting Task" + "\033[0m")
        print(f'  {"Seq Len:":<20}{args.seq_len:<20}{"Label Len:":<20}{args.label_len:<20}')
        print(f'  {"Pred Len:":<20}{args.pred_len:<20}{"Seasonal Patterns:":<20}{args.seasonal_patterns:<20}')
        print(f'  {"Inverse:":<20}{args.inverse:<20}')
        print()

    if args.task_name == 'imputation':
        print("\033[1m" + "Imputation Task" + "\033[0m")
        print(f'  {"Mask Rate:":<20}{args.mask_rate:<20}')
        print()

    if args.task_name == 'anomaly_detection':
        print("\033[1m" + "Anomaly Detection Task" + "\033[0m")
        print(f'  {"Anomaly Ratio:":<20}{args.anomaly_ratio:<20}')
        print()

    print("\033[1m" + "Model Parameters" + "\033[0m")
    print(f'  {"Top k:":<20}{args.top_k:<20}{"Num Kernels:":<20}{args.num_kernels:<20}')
    print(f'  {"Enc In:":<20}{args.enc_in:<20}{"Dec In:":<20}{args.dec_in:<20}')
    print(f'  {"C Out:":<20}{args.c_out:<20}{"d model:":<20}{args.d_model:<20}')
    print(f'  {"n heads:":<20}{args.n_heads:<20}{"e layers:":<20}{args.e_layers:<20}')
    print(f'  {"d layers:":<20}{args.d_layers:<20}{"d FF:":<20}{args.d_ff:<20}')
    print(f'  {"Moving Avg:":<20}{args.moving_avg:<20}{"Factor:":<20}{args.factor:<20}')
    print(f'  {"Distil:":<20}{args.distil:<20}{"Dropout:":<20}{args.dropout:<20}')
    print(f'  {"Embed:":<20}{args.embed:<20}{"Activation:":<20}{args.activation:<20}')
    if hasattr(args, 'use_internal_physics'):
        print(f'  {"Internal Phys:":<20}{args.use_internal_physics:<20}{"Phys Layers:":<20}{args.physics_layers:<20}')
        print(f'  {"Phys Hidden:":<20}{args.physics_hidden_dim:<20}{"Night Cos Thr:":<20}{args.night_cos_threshold:<20}')
        if hasattr(args, 'physics_context_mode'):
            print(f'  {"Context Mode:":<20}{args.physics_context_mode:<20}{"Ctx Gate Init:":<20}{getattr(args, "physics_context_gate_init", "n/a"):<20}')
        if hasattr(args, 'physics_gated_feedback'):
            print(f'  {"Gated Feedback:":<20}{args.physics_gated_feedback:<20}{"Phys Kernel:":<20}{args.physics_kernel_size:<20}')
    if hasattr(args, 'is_three'):
        print(f'  {"Is Three:":<20}{args.is_three:<20}')
    if hasattr(args, 'wind_cap_quantile'):
        if hasattr(args, 'wind_ramp_quantile'):
            print(f'  {"Wind Cap Q:":<20}{args.wind_cap_quantile:<20}{"Wind Ramp Q:":<20}{args.wind_ramp_quantile:<20}')
        else:
            print(f'  {"Wind Cap Q:":<20}{args.wind_cap_quantile:<20}')
    if hasattr(args, 'wind_state_bins'):
        print(f'  {"Wind State Bins:":<20}{args.wind_state_bins:<20}')
    if hasattr(args, 'wind_constraint_weight'):
        print(f'  {"Wind Const W:":<20}{args.wind_constraint_weight:<20}{"Wind Warmup:":<20}{args.wind_constraint_warmup_epochs:<20}')
    if hasattr(args, 'wind_cp_mode'):
        print(f'  {"Wind Cp Mode:":<20}{args.wind_cp_mode:<20}{"Wind Cp Max:":<20}{args.wind_cp_max:<20}')
        print(f'  {"Wind Cp Opt:":<20}{args.wind_cp_optimal_ratio:<20}{"Wind Cp Width:":<20}{args.wind_cp_width:<20}')
    if hasattr(args, 'wind_soft_ramp_weight'):
        print(f'  {"Wind Soft Ramp:":<20}{args.wind_soft_ramp_weight:<20}{"Wind Soft Bound:":<20}{args.wind_soft_bound_weight:<20}')
    if hasattr(args, 'wind_grad_clip'):
        print(f'  {"Wind Grad Clip:":<20}{args.wind_grad_clip:<20}')
    print()

    print("\033[1m" + "Run Parameters" + "\033[0m")
    print(f'  {"Num Workers:":<20}{args.num_workers:<20}{"Itr:":<20}{args.itr:<20}')
    print(f'  {"Train Epochs:":<20}{args.train_epochs:<20}{"Batch Size:":<20}{args.batch_size:<20}')
    print(f'  {"Patience:":<20}{args.patience:<20}{"Learning Rate:":<20}{args.learning_rate:<20}')
    print(f'  {"Des:":<20}{args.des:<20}{"Loss:":<20}{args.loss:<20}')
    print(f'  {"Lradj:":<20}{args.lradj:<20}{"Use Amp:":<20}{args.use_amp:<20}')
    print()

    print("\033[1m" + "GPU" + "\033[0m")
    print(f'  {"Use GPU:":<20}{args.use_gpu:<20}{"GPU:":<20}{args.gpu:<20}')
    print(f'  {"Use Multi GPU:":<20}{args.use_multi_gpu:<20}{"Devices:":<20}{args.devices:<20}')
    print()

    print("\033[1m" + "De-stationary Projector Params" + "\033[0m")
    p_hidden_dims_str = ', '.join(map(str, args.p_hidden_dims))
    print(f'  {"P Hidden Dims:":<20}{p_hidden_dims_str:<20}{"P Hidden Layers:":<20}{args.p_hidden_layers:<20}') 
    print()
