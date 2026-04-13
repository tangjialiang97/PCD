# SolarPhysics 下一步优化计划

## 总目标

在现有 `SolarPhysics` 结果基础上，统一评价口径，明确每个 backbone 的主线方向，并把剩余实验收缩到少量高价值验证项。

---

## 零、基线实验结果（irr，Solar Folsom，无物理层）


| Model        | Pred Len | MSE      | MAE    | NRMSE  | NMAE   |
| ------------ | -------- | -------- | ------ | ------ | ------ |
| DLinear      | 12       | 15823.26 | 94.34  | 0.0858 | 0.0644 |
| DLinear      | 24       | 20970.38 | 113.79 | 0.0988 | 0.0776 |
| DLinear      | 48       | 25566.03 | 129.53 | 0.1091 | 0.0884 |
| DLinear      | 168      | 43522.01 | 174.29 | 0.1423 | 0.1189 |
| iTransformer | 12       | 29602.84 | 140.60 | 0.1174 | 0.0959 |
| iTransformer | 24       | 24487.25 | 126.05 | 0.1067 | 0.0860 |
| iTransformer | 48       | 34472.16 | 150.02 | 0.1266 | 0.1023 |
| iTransformer | 168      | 47637.24 | 181.82 | 0.1489 | 0.1240 |
| PatchTST     | 12       | 70082.23 | 226.11 | 0.1806 | 0.1542 |
| PatchTST     | 24       | 70033.75 | 226.03 | 0.1805 | 0.1542 |
| PatchTST     | 48       | 67067.34 | 221.35 | 0.1767 | 0.1510 |
| PatchTST     | 168      | 67279.66 | 221.56 | 0.1769 | 0.1511 |


---

## 一、统一结论

### 1.1 统一口径

- “当前最低分”指现有结果文件里已经出现过的最好三通道结果，不强行区分是否使用辅助项。
- “主线推荐”指后续最值得继续投入的 backbone 与配置方向，优先考虑：
  - 可解释性
  - 对内部物理层本身的贡献判断
  - 后续继续做结构增强的空间

### 1.2 当前最稳的全局结论

- 三通道口径下，`SolarPhysics` 的收益不是只来自 `GHI`，`DHI` 的改善同样明显，`DNI` 仍是主要瓶颈。
- 辅助项不是统一默认增益项：
  - 中短期部分组合受 `limit_only / var_only` 帮助明显
  - 长期尤其 `TimeMixerSolarPhysics@168` 对辅助项或更强物理约束更敏感
- `night_cos_threshold` 不是全局参数：
  - 默认仍应以 `0.0` 为主
  - 只有 `TimeMixerSolarPhysics@168` 明确受益于 `0.1`
- `SolarPhysicsBlock` 当前只保留一个明确正向方向：`gated feedback`
- `state residual update` 当前实现不稳定，应暂停
- `dual_branch` 暂未看到稳定收益，不应继续作为主线
- 简化物理上界不是统一增益项：
  - `TimeMixerSolarPhysics` 明显受益
  - `iTransformerSolarPhysics` 只有小幅收益
  - `PatchTSTSolarPhysics / DLinearSolarPhysics` 未见稳定收益
- 在第一阶段搜索里，最稳定、最值得继续保留的主效应是：
  - `physics_layers`
  - `night_cos_threshold`

### 1.3 纯基线 vs SolarPhysicsBlock vs SolarPhysicsBlock+aux 对照表

> 统一按单通道 `NRMSE` 对比，数值越低越好。  
> `baseline` = 纯 backbone，不使用 `SolarPhysicsBlock`  
> `SP` = 使用 `SolarPhysicsBlock`  
> `SP+aux` = 使用 `SolarPhysicsBlock + auxiliary`  
> `SP+aux(avg_3ch)` = 全部损失约束下，对应实验的 `avg_3ch_nrmse`  
> `TimeMixer` 当前缺少纯基线对照，因此该列记为 `-`


| Model          | pred_len | baseline  | SP        | SP+aux    | SP+aux(avg_3ch) | 最优配置     | 结论                                        |
| -------------- | -------- | --------- | --------- | --------- | --------------- | -------- | ----------------------------------------- |
| `PatchTST`     | 12       | `0.06275` | `0.03969` | `0.03617` | `0.08858`       | `SP+aux` | `SolarPhysicsBlock` 本身已显著有效，`aux` 继续小幅增益  |
| `PatchTST`     | 24       | `0.07055` | `0.06716` | `0.06248` | `0.11784`       | `SP+aux` | 物理层单独提升有限，进一步收益主要来自 `aux`                 |
| `PatchTST`     | 48       | `0.08226` | `0.06317` | `0.03892` | `0.09086`       | `SP+aux` | `aux` 增益明显，这一档更依赖辅助项                      |
| `PatchTST`     | 168      | `0.19264` | `0.03579` | `0.06761` | `0.11532`       | `SP`     | 这一档最强信号来自内部物理层本身，`aux` 反而破坏结果             |
| `DLinear`      | 12       | `0.08822` | `0.06263` | `0.06531` | `0.11670`       | `SP`     | `SolarPhysicsBlock` 本身有效，当前这组 `aux` 略有副作用 |
| `DLinear`      | 24       | `0.10069` | `0.03702` | `0.03991` | `0.09072`       | `SP`     | 最优结果主要来自内部物理层本身                           |
| `DLinear`      | 48       | `0.11457` | `0.03627` | `0.03950` | `0.09078`       | `SP`     | 最优结果主要来自内部物理层本身                           |
| `DLinear`      | 168      | `0.15366` | `0.06227` | `0.06467` | `0.11317`       | `SP`     | 长期同样以内部物理层贡献为主，`aux` 未带来额外收益              |
| `iTransformer` | 12       | `0.14555` | `0.06157` | `0.03668` | `0.08284`       | `SP+aux` | 物理层先带来大幅提升，`aux` 再继续显著优化                  |
| `iTransformer` | 24       | `0.14525` | `0.06364` | `0.06172` | `0.11853`       | `SP+aux` | 物理层是主贡献，`aux` 只做小幅修正                      |
| `iTransformer` | 48       | `0.13316` | `0.03708` | `0.03630` | `0.08748`       | `SP+aux` | 物理层已经很强，`aux` 只带来边际收益                     |
| `iTransformer` | 168      | `0.17784` | `0.06612` | `0.03663` | `0.08076`       | `SP+aux` | 长期结果对 `aux` 很敏感，提升非常明显                    |


| Model          | pred_len | baseline  | 损失优化  | 
| -------------- | -------- | --------- | --------- |
| `PatchTST`     | 12       | `0.06275` | `0.06274` | 
| `PatchTST`     | 24       | `0.07055` | `0.07249` |
| `PatchTST`     | 48       | `0.08226` | `0.0822` | 
| `PatchTST`     | 168      | `0.19264` | `0.1908` | 
| `iTransformer` | 12       | `0.14555` | `0.1326` | 
| `iTransformer` | 24       | `0.14525` | `0.1073` | 
| `iTransformer` | 48       | `0.13316` | `0.1331` | 
| `iTransformer` | 168      | `0.17784` | `0.1700` |

**表后结论**

- `SolarPhysicsBlock` 对 `PatchTST / DLinear / iTransformer` 都带来了明确正收益。
- `DLinear` 最能证明“内部物理层本身有效”，因为 `24 / 48 / 168` 都是 `SP` 优于 `SP+aux`。
- `PatchTST` 呈现明显 horizon 依赖：
  - `12 / 24 / 48` 更适合 `SP+aux`
  - `168` 明确以 `SP` 最优
- `iTransformer` 是三类实验里最均衡的一组：
  - `SP` 本身已经很强
  - `SP+aux` 在各 horizon 上通常还能继续提升
- `TimeMixer` 当前最依赖配置选择：
  - `168` 上 `SP+aux` 很强
  - `48` 上 `aux` 明显有害
  - 还需要补纯基线后再做完整归因

### 1.4 当前推荐总表


| Horizon | 当前最低分结果                                                      | 主线推荐                                                       | 是否依赖辅助项 | 统一判断                                                    |
| ------- | ------------------------------------------------------------ | ---------------------------------------------------------- | ------- | ------------------------------------------------------- |
| `12`    | `TimeMixerSolarPhysics + limit_only`，`avg_3ch_nrmse≈0.08057` | `PatchTSTSolarPhysics + gated feedback`                    | 中等      | 若只看分数，`TimeMixer` 有竞争力；若看内部物理层可控性与后续结构增强，仍优先 `PatchTST` |
| `24`    | `DLinearSolarPhysics`，最好约 `avg_3ch_nrmse≈0.07929`            | `DLinearSolarPhysics`                                      | 低       | 当前最稳的轻量强基线，适合作为所有后续改动的比较对象                              |
| `48`    | `iTransformerSolarPhysics`，最好约 `avg_3ch_nrmse≈0.07927`       | `iTransformerSolarPhysics + gated feedback`                | 低到中     | 这一档最适合继续做层内结构增强                                         |
| `168`   | `TimeMixerSolarPhysics + aux`，约 `avg_3ch_nrmse≈0.08005`      | `TimeMixerSolarPhysics + night_cos_threshold=0.1`，并继续联调弱上界 | 高       | 长 horizon 目前最值得投入的主线                                    |


### 1.5 收益来源的统一判断

- 可较明确归因为“内部物理层本身贡献”的信号主要来自：
  - `PatchTSTSolarPhysics@168`
  - `DLinearSolarPhysics@24`
  - `DLinearSolarPhysics@48`
  - 部分 `iTransformer / TimeMixer` 中短期组合
- 大幅提升约 `40%` 的组合多数仍伴随 `limit_only / var_only / aux`，因此不能直接归因为内部物理层结构本身。
- 后续所有“最好结果”都应同时标注：
  - 是否使用 auxiliary
  - 是否属于内部物理层主贡献

### 1.6 当前已经确认的结构结论

- `PatchTSTSolarPhysics@12`
  - baseline：`avg_3ch_nrmse=0.08539`
  - `gated feedback`：降到 `0.08371`
  - 第一阶段搜索最好约 `0.08161`
- `DLinearSolarPhysics@24`
  - baseline：`avg_3ch_nrmse=0.08046`
  - 第一阶段搜索最好约 `0.07929`
  - 当前仍是最稳的轻量基线
- `iTransformerSolarPhysics@48`
  - baseline：`avg_3ch_nrmse=0.08672`
  - `gated feedback`：降到 `0.08175`
  - 第一阶段搜索最好约 `0.07927`
- `TimeMixerSolarPhysics@168`
  - baseline：`avg_3ch_nrmse=0.12128`
  - `night_cos_threshold=0.1`：降到 `0.08418`
  - 弱上界最好约 `0.08260`
  - `_True` 文件中最好约 `0.08005`

### 1.7 统一实验设置

**统一目标**

- 后续所有主线实验都基于同一套默认配置启动
- 每次只改少量关键变量，避免多因素同时变化导致结论混乱

**公共默认配置**

- 数据集：`Solar / Folsom_combined_no_zero.csv`
- 任务：`long_term_forecast`
- 输入输出通道：`enc_in=3, dec_in=3, c_out=3`
- 特征设置：`features=M`
- 长度设置：`seq_len=256, label_len=0`
- 评估设置：`is_three=True`
- 训练轮次：`itr=1`
- 基础结构参数：`d_model=128, d_ff=128, e_layers=2, d_layers=1, factor=3`
- 物理层默认：`physics_layers=2, physics_hidden_dim=128, physics_kernel_size=3, physics_eps=1e-4`
- 太阳几何默认：`night_cos_threshold=0.0`
- 块增强默认：仅允许 `physics_gated_feedback`
- 简化物理上界默认：关闭
- 辅助项默认：关闭

**主线实验的模型级覆盖项**

- `PatchTSTSolarPhysics@12`
  - 默认开启：`physics_gated_feedback`
  - 保持：`night_cos_threshold=0.0`
  - 默认关闭：`weak upper bound`, `auxiliary`
- `DLinearSolarPhysics@24`
  - 作为基线：不启用额外结构增强
  - 保持：`night_cos_threshold=0.0`
  - 默认关闭：`weak upper bound`, `auxiliary`
- `iTransformerSolarPhysics@48`
  - 默认开启：`physics_gated_feedback`
  - 保持：`night_cos_threshold=0.0`
  - `weak upper bound` 仅作为局部联调项，不作为默认项
- `TimeMixerSolarPhysics@168`
  - 默认改为：`night_cos_threshold=0.1`
  - 允许继续测试：`weak upper bound`
  - `auxiliary` 可作为长期主线的一部分保留

**统一调参原则**

- 每轮只改一个主效应：
  - `physics_layers`
  - `night_cos_threshold`
  - `physics_hidden_dim`
  - `weak upper bound`
  - `use_auxiliary`
- 若做联合搜索，只允许在当前主线模型上进行
- 所有非主线模型都回退到公共默认配置

**统一结果记录规范**

- 每条结果必须记录：
  - 模型名
  - `pred_len`
  - 是否使用 `auxiliary`
  - 是否开启 `gated feedback`
  - 是否开启 `weak upper bound`
  - `night_cos_threshold`
  - `physics_layers / physics_hidden_dim / physics_kernel_size`
- 每条结果至少保留：
  - `avg_3ch_mse`
  - `avg_3ch_nrmse`
  - `GHI / DNI / DHI` 各自 `nrmse`
- 每个 horizon 最终要同时标注：
  - `score-best`
  - `internal-physics-best`
  - `needs-aux`

---

## 二、模型分线计划

### 2.1 `PatchTSTSolarPhysics`

**定位**

- 短期主线 backbone
- 适合继续做局部结构增强

**当前结论**

- `12` 步上内部物理层已经可用
- `gated feedback` 有正向收益
- `weak upper bound` 未见稳定增益
- `night_cos_threshold` 应保持 `0.0`

**后续主线**

- 固定 `night_cos_threshold=0.0`
- 保留 `gated feedback`
- 继续围绕 `physics_layers / physics_hidden_dim / physics_kernel_size` 做小范围细化

**暂缓项**

- 暂缓 `state residual`
- 暂缓继续单独搜索 `weak upper bound`
- 暂缓 `dual_branch`

### 2.2 `DLinearSolarPhysics`

**定位**

- 轻量强基线
- 用于验证新物理层改动是否真的带来收益

**当前结论**

- `24` 步结果最稳
- 多组结果说明即使不依赖辅助项，内部物理层本身也能工作
- `gated / dual_branch / weak upper bound` 都没有形成稳定正收益

**后续主线**

- 作为对照 baseline 长期保留
- 只做少量低成本局部搜索，重点放在：
  - `physics_layers`
  - `physics_hidden_dim`

**暂缓项**

- 不把 `DLinear` 作为结构创新主战场
- 暂不继续投入 `weak upper bound` 与 block 复杂增强

### 2.3 `iTransformerSolarPhysics`

**定位**

- 中期主线 backbone
- 当前最适合继续做层内增强的模型之一

**当前结论**

- `48` 步是最强档位之一
- `gated feedback` 明确有效
- `weak upper bound` 有小幅收益，但不是主效应
- `night_cos_threshold` 仍应保持 `0.0`

**后续主线**

- 固定 `night_cos_threshold=0.0`
- 优先保留 `gated feedback`
- 继续局部联调：
  - `physics_layers`
  - `physics_hidden_dim`
  - `weak upper bound`

**暂缓项**

- 暂缓 `state residual`
- `dual_branch` 只保留为低优先级验证项，不再放在主线

### 2.4 `TimeMixerSolarPhysics`

**定位**

- 长期主线 backbone
- 当前最依赖“按模型定制物理层”的组合

**当前结论**

- `168` 步改善空间最大
- `night_cos_threshold=0.1` 是明确有效信号
- `weak upper bound` 明显有效
- `_True` 结果也表明该模型对更强物理约束或辅助配置更敏感

**后续主线**

- 固定优先搜索区间：
  - `night_cos_threshold=0.05 / 0.1 / 0.15`
  - `weak upper bound`
  - `physics_layers`
- 允许继续保留 `auxiliary` 作为高 horizon 方案的一部分

**暂缓项**

- 在补齐更关键的联合搜索前，不优先扩展 `dual_branch`

### 2.5 当前建议保留的模型组合

- `PatchTSTSolarPhysics + gated feedback`
- `DLinearSolarPhysics` 作为轻量基线
- `iTransformerSolarPhysics + gated feedback`
- `TimeMixerSolarPhysics + night_cos_threshold=0.1`
- `TimeMixerSolarPhysics + weak upper bound`

---

## 三、待验证项

### 3.1 高优先级

- 把 `future_zenith` 从标签列读取改为按未来时间现算，消除实现歧义。
- 建立统一实验汇总脚本，统一输出：
  - 每个 `pred_len` 的 top-k
  - 是否使用 auxiliary
  - 三通道指标
  - 白天 closure 误差
  - 夜间非零 `GHI` 指标
- 对所有“当前最好结果”补统一标签：
  - `score-best`
  - `internal-physics-best`
  - `needs-aux`

### 3.2 中优先级

- `TimeMixerSolarPhysics@168` 联合搜索：
  - `night_cos_threshold`
  - `weak upper bound`
  - `use_auxiliary`
- `iTransformerSolarPhysics@48` 联合搜索：
  - `gated feedback`
  - `physics_layers`
  - `weak upper bound`
- `PatchTSTSolarPhysics@12` 继续局部搜索：
  - `gated feedback`
  - `physics_layers`
  - `physics_hidden_dim`

### 3.3 低优先级

- 补做 `iTransformerSolarPhysics@48 + dual_branch`，仅用于彻底关掉这条分支。
- 补做 `TimeMixerSolarPhysics@168` 的 block 对照实验，但仅在不影响主线搜索时再做。
- 继续小范围验证 `physics_kernel_size`，但不再作为主效应主线。

### 3.4 推荐执行顺序

1. 统一结果汇总脚本与口径。
2. 修正 `future_zenith` 的实现来源。
3. 继续 `PatchTST@12 / iTransformer@48 / TimeMixer@168` 三条主线。
4. 保留 `DLinear@24` 作为所有新增改动的对照基线。
5. 只在主线收敛后，再回头补低优先级验证项。

### 3.5 完成标志

- 已能清楚区分“最低分结果”和“内部物理层主贡献结果”。
- 已形成统一的三通道实验汇总表。
- 已确认各 horizon 的主线 backbone。
- 已完成 `PatchTST@12 / iTransformer@48 / TimeMixer@168` 的主线局部收敛。
- 已确认哪些物理层增强值得保留，哪些方向正式暂停。

---


## cmpSeq24Pl1 结果表

| Model | Config | avg_3ch_mse | avg_3ch_mae | avg_3ch_nrmse | avg_3ch_nmae | GHI_nrmse | DNI_nrmse | DHI_nrmse |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PatchTST | baseline | 6274.1294 | 36.1230 | 0.06633 | 0.03364 | 0.04090 | 0.11950 | 0.03861 |
| PatchTST | SP | 7063.1348 | 48.2350 | 0.07232 | 0.04697 | 0.03802 | 0.12857 | 0.05036 |
| PatchTST | SP+aux | 12411.6953 | 79.6864 | 0.11137 | 0.08400 | 0.05105 | 0.14331 | 0.13975 |
| DLinear | baseline | 7112.0444 | 44.9274 | 0.06996 | 0.04153 | 0.04539 | 0.12637 | 0.03812 |
| DLinear | SP | 4559.2842 | 32.0455 | 0.05698 | 0.03058 | 0.03278 | 0.10290 | 0.03525 |
| DLinear | SP+aux | 11068.0010 | 76.0207 | 0.10660 | 0.07967 | 0.05427 | 0.12577 | 0.13975 |
| iTransformer | baseline | 9722.6914 | 57.0012 | 0.08305 | 0.05161 | 0.06004 | 0.14137 | 0.04776 |
| iTransformer | SP | 14075.6328 | 82.8243 | 0.11731 | 0.08537 | 0.05637 | 0.15580 | 0.13975 |
| iTransformer | SP+aux | 13667.0791 | 81.8685 | 0.11607 | 0.08446 | 0.05764 | 0.15083 | 0.13975 |

## 精简版

| Model | baseline NRMSE | SP NRMSE | SP+aux NRMSE | Best |
| --- | ---: | ---: | ---: | --- |
| PatchTST | 0.06633 | 0.07232 | 0.11137 | baseline |
| DLinear | 0.06996 | 0.05698 | 0.10660 | SP |
| iTransformer | 0.08305 | 0.11731 | 0.11607 | baseline |
| TimeMixer | - | 0.05700 | 0.05773 | SP |