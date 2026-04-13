import pandas as pd
import re
import numpy as np

def process_forecast_data(txt_path, csv_path, output_path):
    # 1. 解析 txt 文件中的数据
    extracted_data = {}
    
    # 正则表达式匹配配置行 (例如: ..._256_12_iTransformer_Solar...)
    # 提取第3部分的数字作为长度 (12)，第4部分作为模型名 (iTransformer)
    config_pattern = re.compile(r"long_term_forecast_stanford\.csv_\d+_(\d+)_([A-Za-z0-9]+)_")
    # 正则表达式匹配指标行
    metrics_pattern = re.compile(r"nrmse:([\d\.]+),\s*nmae:([\d\.]+)")
    
    current_model = None
    current_len = None
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 检查是否是配置行
        config_match = config_pattern.search(line)
        if config_match:
            current_len = config_match.group(1)
            current_model = config_match.group(2)
            continue
            
        # 检查是否是指标行
        metrics_match = metrics_pattern.search(line)
        if metrics_match and current_model and current_len:
            nrmse = float(metrics_match.group(1))
            nmae = float(metrics_match.group(2))
            
            # 存入字典，key为 (模型名, 长度)
            extracted_data[(current_model, current_len)] = (nrmse, nmae)
            
            current_model = None # 重置
            current_len = None

    # 2. 读取 CSV/Excel 模板
    # 注意：根据你上传的文件，这是一个CSV文件
    df = pd.read_csv(csv_path)
    
    # 3. 处理表格结构并填充数据
    # 创建一个临时列，填充合并单元格造成的空缺（向下填充模型名）
    # 同时去除可能存在的空格
    methods_series = df['方法'].fillna(method='ffill').astype(str).str.strip()
    
    # 获取需要计算平均值的长度列表
    valid_lengths = ['12', '24', '48', '168']
    
    # 遍历表格行进行填充
    for i in range(1, len(df)): # 跳过表头第一行
        method = methods_series.iloc[i]
        length_val = df.iloc[i, 1] # '预测长度'列
        
        # 确保长度是字符串格式且无空格
        length = str(length_val).strip()
        
        if length == 'avg':
            continue #稍后计算
            
        # 查找提取的数据中是否有对应值
        key = (method, length)
        if key in extracted_data:
            nrmse, nmae = extracted_data[key]
            # 填入 Standford 对应的列 (索引4和5)
            df.iloc[i, 4] = nrmse
            df.iloc[i, 5] = nmae

    # 4. 计算并填充 avg 行
    unique_methods = methods_series.unique()
    
    for method in unique_methods:
        if method == 'nan': continue
        
        # 找到该模型对应的所有行索引
        method_indices = methods_series[methods_series == method].index
        
        nrmse_vals = []
        nmae_vals = []
        avg_row_idx = -1
        
        for idx in method_indices:
            length = str(df.iloc[idx, 1]).strip()
            
            # 收集有效长度的数据
            if length in valid_lengths:
                val_nrmse = df.iloc[idx, 4]
                val_nmae = df.iloc[idx, 5]
                
                if pd.notna(val_nrmse): nrmse_vals.append(float(val_nrmse))
                if pd.notna(val_nmae): nmae_vals.append(float(val_nmae))
            
            # 标记 avg 行的位置
            elif length == 'avg':
                avg_row_idx = idx
        
        # 如果找到了 avg 行且有数据，进行计算填充
        if avg_row_idx != -1:
            if nrmse_vals:
                df.iloc[avg_row_idx, 4] = np.mean(nrmse_vals)
            if nmae_vals:
                df.iloc[avg_row_idx, 5] = np.mean(nmae_vals)

    # 5. 保存结果
    # 可以保存为Excel格式，方便查看
    df.to_excel(output_path, index=False)
    print(f"处理完成，文件已保存至: {output_path}")

# 使用示例 (请确保文件名与你本地的一致)
txt_file = 'result_long_term_forecast_Solar_folsom.txt'
csv_file = 'Folsom_irradiance_train.csv'
output_file = 'filled_data_folsom.xlsx'

process_forecast_data(txt_file, csv_file, output_file)