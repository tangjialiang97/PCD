# TimeMoE 版本兼容性修复指南

## 问题描述
`AttributeError: 'TimeMoeForPrediction' object has no attribute '_extract_past_from_model_output'`

这个错误是由于 transformers 库版本过新导致的。在 transformers 4.41.0 及以上版本中，`DynamicCache` 类移除了 `seen_tokens` 属性，导致 TimeMoE 模型的自定义生成代码无法正常工作。

## 解决方案

### 方案1：降级 transformers 版本（推荐）

将 transformers 降级到 4.40.0 或更早版本：

```bash
# 降级到 4.40.0（推荐）
pip install transformers==4.40.0

# 或者降级到 4.36.2（更保守的选择）
pip install transformers==4.36.2
```

### 方案2：使用代码修复（已实现）

代码中已经添加了兼容性修复，但如果降级版本更简单，建议使用方案1。

## 验证修复

运行以下命令验证版本：

```bash
python -c "import transformers; print(f'transformers version: {transformers.__version__}')"
```

然后重新运行您的脚本：

```bash
bash scripts/long_term_forecast/ETT_script/LTSM.sh
```

## 注意事项

1. 降级 transformers 可能会影响其他依赖该库的代码
2. 如果项目中有其他模型需要新版本的 transformers，可能需要使用虚拟环境隔离
3. 建议在 requirements.txt 中固定 transformers 版本以避免未来出现类似问题

