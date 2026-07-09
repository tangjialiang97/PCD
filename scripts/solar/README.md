# GHI → 发电功率集中映射方案

本文说明本仓库中 **辐照（GHI）→ 发电功率（power）** 的完整方案：  
需要哪些变量、离线如何造表、模型内如何映射、以及怎么跑通实验。

相关代码入口：

| 路径 | 作用 |
|------|------|
| `scripts/solar/build_plant_power_dataset.py` | 真实电站 + 气象 → 离线 `power_pvlib` 等列 |
| `scripts/solar/build_power_dataset.py` | Congo 等仅 GHI 数据 → 合成 power 标签 |
| `models/_solar_physics_utils.py` | 模型内 `SolarPowerMap` + direct power + fusion |
| `scripts/solar/plant_power.sh` | 电站功率一键流程 |
| `scripts/solar/power.sh` | Congo / 合成功率实验 |
| `run.py` | 训练入口与全部 CLI |

---

## 1. 总体流程

```text
                    ┌─────────────────────────────┐
  历史 GHI/DNI/DHI  │  时序 backbone + SolarPhysics │
  (/power 可选)     │  可行集投影 → 预测 GHI        │
  (+ zenith)        │                             │
                    └──────────────┬──────────────┘
                                   │ projected GHI
                                   ▼
                    ┌─────────────────────────────┐
                    │  SolarPowerMap              │
                    │  prior: linear | pvwatts    │
                    │  head:  fixed | affine |    │
                    │         residual_mlp |      │
                    │         learned | ghi_mlp   │
                    └──────────────┬──────────────┘
                                   │ mapped power
       direct power（第4通道预测；无第4通道时用 fallback head）
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │  SolarPowerFusion           │
                    │  mapped/direct/average/     │
                    │  gated/learned              │
                    └──────────────┬──────────────┘
                                   │
                              最终 power
                                   │
                     监督标签：observe_power
                     （或合成 power / power_pvlib 分析）
```

两层含义不要混：

1. **离线列**：用 pvlib 在 CSV 里写出 `power_pvlib`，用于分析、标定 `pdc0`、对照。  
2. **模型内映射**：训练/推理时，在投影 GHI 后用可微 prior + 可选残差头得到 mapped power。
3. **最终融合**：mapped power 和直接预测的 power 通过 `--solar_power_fusion` 得到最终 power。

推荐组合：**离线 pvlib 造表 + 模型内 `pvwatts` prior + `residual_mlp` 残差 + `learned`/`gated` 融合**，标签用真实 `observe_power`。

---

## 2. 需要哪些变量

### 2.1 真实电站数据（推荐）

**电站表（必需）**

| 列名 | 含义 | 用途 |
|------|------|------|
| `dtime` | 时间戳 | 对齐、太阳位置、序列索引 |
| `observe_ghi` | 观测 GHI | 映射主输入 / 模型 `ghi` |
| `observe_power` | 观测功率 | **训练与评估标签 `power`** |

**气象 / 预报表（强烈建议，可缺）**

| 列名 | 含义 | 用途 |
|------|------|------|
| `dtime` | 时间戳 | 与电站 `merge_asof` |
| `GHI` | 气象/预报 GHI | 交叉检验、预报任务外生输入 |
| `TEMP` | 气温 | Faiman/NOCT 估电池温度 \(T_c\) |
| `WS` | 风速 | Faiman 冷却项 |
| `WD` | 风向 | 可选残差特征（不进 pvlib 核心） |
| `PREC` | 降水 | 可选残差特征 |
| `PWAT` | 可降水量 | 可选残差特征 |
| `SDWE` | 雪水当量等 | 可选残差特征 |
| `interval` / `interval_day` | 预报时效 | 4h / 1d 任务对齐 |
| `report_date` / `report_time` | 发布时间 | lead time |
| `update_time` | 更新时间 | 数据版本 |
| `govern_flag` | 质控标记 | 过滤坏点 |

**站点元数据（必需，可写在命令行）**

| 参数 | 示例 | 用途 |
|------|------|------|
| `latitude` | `29.919` | 太阳位置 / cos zenith |
| `longitude` | `100.641` | 同上 |
| `timezone` | `Asia/Shanghai` | 无时区时间戳本地化 |

### 2.2 最小训练 CSV

如果数据集中已经有真实发电量，最小列可以只有：

```text
dtime, ghi, dni, dhi, power
```

读取时会自动按 `ghi,dni,dhi,power` 排列。`SolarPhysics` 仍只对前三列做物理闭包投影；当 `enc_in=4` 时，第 4 列 `power` 会进入 backbone，backbone 的第 4 个输出作为直接预测 power 分支。

**可选配置（无则自动/默认）**

| 参数 | 默认 | 说明 |
|------|------|------|
| `pdc0` / capacity | 白天 `median(P/(GHI/1000))` 自动估 | 装机尺度 |
| `gamma_pdc` | `-0.004` | PVWatts 温度系数 (1/°C) |
| `temp_ref` | `25` | 参考电池温度 (°C) |
| 倾角/方位 | 未知则水平近似 | 当前不做斜面转置 |
| DNI/DHI | 可用 Erbs 从 GHI 估计 | 仅为兼容 `enc_in=3` |

### 2.2 Congo 类仅辐照数据（合成标签）

| 列名 | 含义 |
|------|------|
| `timeStamp` | 时间 |
| `ghi, dni, dhi` | 三通道辐照 |

无真实功率时，用 `build_power_dataset.py` 合成 `power`（`linear` 或 `nonlinear`）。

### 2.3 训练 CSV 统一列（电站流程输出）

`build_plant_power_dataset.py` 写出例如：

```text
timeStamp, ghi, dni, dhi, power, power_pvlib,
temp_air, wind_speed, temp_cell, cos_zenith, pdc0,
[power_linear], [weather extras...]
```

| 列 | 含义 |
|----|------|
| `ghi` | `observe_ghi` |
| `power` | **真实标签** = `observe_power` |
| `power_pvlib` | 离线 Faiman + `pvlib.pvsystem.pvwatts_dc` |
| `dni, dhi` | Erbs 估计（可选，给 SolarPhysics 用） |
| `temp_* / wind_speed / cos_zenith / pdc0` | 分析与标定 |

---

## 3. 映射公式

### 3.1 线性 prior

\[
P_{\mathrm{linear}} = C \cdot \frac{\max(G_{\mathrm{GHI}}, 0)}{G_{\mathrm{ref}}}
\]

默认 \(G_{\mathrm{ref}}=1000\)。

### 3.2 PVWatts prior（与 pvlib 对齐）

电池温度（二选一）：

- **NOCT**（模型默认）：\(T_c = T_a + G/800\cdot(\mathrm{NOCT}-20)\)
- **Faiman**（离线默认）：\(T_c = T_a + G/(u_0+u_1 W_s)\)

直流功率：

\[
P_{\mathrm{pvwatts}} = C \cdot \frac{G}{G_{\mathrm{ref}}} \cdot \bigl(1 + \gamma_{pdc}(T_c - T_{\mathrm{ref}})\bigr)_+
\]

### 3.3 模型映射头（加在 prior 上）

| `--solar_power_map` | 公式 |
|---------------------|------|
| `fixed` | \(P = \mathrm{prior}\) |
| `affine` | \(P = \mathrm{softplus}(a)\cdot\mathrm{prior} + b\) |
| `residual_mlp` | \(P = \mathrm{prior} + \mathrm{MLP}([g_n, p_n, \cos z])\)（末层零初始化） |
| `learned` | \(P = \mathrm{softplus}(\mathrm{MLP}([g_n,\cos z]))\cdot C\)（不加 prior） |
| `ghi_mlp` | \(P = \mathrm{softplus}(\mathrm{MLP}([g_n]))\cdot C\)（严格只用 GHI） |

其中 \(g_n=G/G_{\mathrm{ref}}\)，\(p_n=\mathrm{prior}/C\)。

### 3.4 直接 power 分支与最终融合

模型最终监督的是：

\[
P_{\mathrm{final}}=\mathrm{Fusion}(P_{\mathrm{map}}, P_{\mathrm{direct}})
\]

- \(P_{\mathrm{map}}\)：先通过物理模型得到可行集内的 `ghi`，再用 `--solar_power_map` 映射得到。
- \(P_{\mathrm{direct}}\)：当 `enc_in=4` 且输入包含 `power` 时，使用 backbone 的第 4 个输出；当 `enc_in=3` 时，使用 fallback direct head。

| `--solar_power_fusion` | 含义 |
|------------------------|------|
| `mapped` | 只用 \(P_{\mathrm{map}}\)（兼容旧逻辑） |
| `direct` | 只用 \(P_{\mathrm{direct}}\) |
| `average` | 两者平均 |
| `gated` | 学一个全局门控 \(\alpha P_{\mathrm{map}}+(1-\alpha)P_{\mathrm{direct}}\) |
| `learned` | 根据 `[P_map, P_direct, ghi_n, cos_z]` 学逐时刻门控 |

`--solar_power_map ghi_mlp` 是“只从 GHI 学映射到 power”的方案；它同样可以再配 `--solar_power_fusion gated/learned` 与直接预测 power 结合。

---

## 4. 怎么运行

### 4.0 环境

```bash
# 仓库根目录
cd /path/to/PCD
# 需已安装 pandas / numpy / pvlib / torch 等（见 requirements.txt）
# 电站 xlsx 需要 openpyxl：pip install openpyxl
```

### 4.1 路径 B：真实电站 + 离线 pvlib + 模型残差（推荐）

**Step 1 — 造训练表**

```bash
python scripts/solar/build_plant_power_dataset.py \
  --plant /path/to/plant.xlsx \
  --weather /path/to/weather.xlsx \
  --latitude 29.919 \
  --longitude 100.641 \
  --timezone Asia/Shanghai \
  --temp-cell-method faiman \
  --add-erbs-components True \
  --include-linear-prior True \
  --output data/solar/Qingda_plant_power.csv
```

无气象文件时可去掉 `--weather`（\(T_a,W_s\) 用默认值）。

终端会打印：

- 自动 `pdc0`
- `MAE(power_pvlib, observe_power)` 与相关系数  

请记下 `pdc0`，训练时传给 `--solar_power_capacity`。

**Step 2 — 训练（四通道输出 + PVWatts prior + 残差 MLP + learned 融合）**

```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/solar/ \
  --data_path Qingda_plant_power.csv \
  --model DLinearSolarPhysics \
  --data Solar \
  --features M \
  --seq_len 256 \
  --pred_len 24 \
  --label_len 0 \
  --batch_size 256 \
  --enc_in 4 --dec_in 4 --c_out 4 \
  --d_model 128 --d_ff 128 \
  --des PlantPower \
  --solar_power_output append \
  --solar_power_map residual_mlp \
  --solar_power_prior pvwatts \
  --solar_power_fusion learned \
  --solar_power_capacity <上一步的pdc0> \
  --solar_power_ghi_ref 1000.0 \
  --solar_power_gamma_pdc -0.004 \
  --solar_power_temp_cell_method noct \
  --solar_power_temp_amb 20.0 \
  --gpu 0
```

此时模型输出监督列为 `[ghi,dni,dhi,power]`，最终 `power` 是 \(P_{\mathrm{map}}\) 和 \(P_{\mathrm{direct}}\) 的融合结果。若只训练/评估发电量，把 `--solar_power_output append` 改为 `power` 即可。

**或一键脚本**（先改脚本内 `plant_table` / `weather_table`）：

```bash
bash scripts/solar/plant_power.sh
```

### 4.2 路径 A：Congo 合成功率

```bash
# 合成非线性 power 列
python scripts/solar/build_power_dataset.py \
  --input data/solar/Congo.csv \
  --output data/solar/Congo_power.csv \
  --label-mode nonlinear

# 训练（默认 linear prior + residual_mlp；可 export prior_mode=pvwatts）
bash scripts/solar/power.sh
# 或：
prior_mode=pvwatts bash scripts/solar/power.sh
```

### 4.3 只评估离线 prior（不训练）

造完 CSV 后直接比较：

```bash
python - <<'PY'
import pandas as pd
import numpy as np
df = pd.read_csv('data/solar/Qingda_plant_power.csv')
m = np.isfinite(df['power']) & np.isfinite(df['power_pvlib'])
print('MAE', np.mean(np.abs(df.loc[m,'power']-df.loc[m,'power_pvlib'])))
print('corr', np.corrcoef(df.loc[m,'power'], df.loc[m,'power_pvlib'])[0,1])
print(df[['ghi','power','power_pvlib','temp_cell']].describe())
PY
```

---

## 5. CLI 速查

### 5.1 `build_plant_power_dataset.py`

| 参数 | 默认 | 说明 |
|------|------|------|
| `--plant` | 必填 | 电站表 xlsx/csv |
| `--weather` | 空 | 气象/预报表 |
| `--output` | `data/solar/plant_power.csv` | 输出路径 |
| `--latitude` / `--longitude` | `29.919` / `100.641` | 站点 |
| `--timezone` | `Asia/Shanghai` | 时区 |
| `--merge-tolerance` | `30min` | 天气 asof 对齐容差 |
| `--pdc0` | 自动 | 装机；不设则白天中位数估计 |
| `--ghi-ref` | `1000` | 参考辐照 |
| `--gamma-pdc` | `-0.004` | 温度系数 |
| `--temp-cell-method` | `faiman` | `faiman` \| `noct` |
| `--temp-amb` / `--wind-default` | `20` / `1` | 缺测回退 |
| `--add-erbs-components` | `True` | 写 `dni,dhi` |
| `--include-linear-prior` | `True` | 写 `power_linear` |
| `--govern-ok` | 空 | 只保留该 `govern_flag` |

### 5.2 `run.py` 功率相关

| 参数 | 默认 | 说明 |
|------|------|------|
| `--solar_power_output` | `irradiance` | `irradiance` \| `power` \| `append` |
| `--solar_power_map` | `fixed` | `fixed` \| `affine` \| `residual_mlp` \| `learned` \| `ghi_mlp` |
| `--solar_power_prior` | `linear` | `linear` \| `pvwatts` |
| `--solar_power_capacity` | `1.0` | 对应 \(C\) / `pdc0` |
| `--solar_power_ghi_ref` | `1000` | \(G_{\mathrm{ref}}\) |
| `--solar_power_clip` | `False` | 是否截断 \(G/G_{\mathrm{ref}}\le 1\) |
| `--solar_power_map_hidden` | `16` | 残差/纯学习 MLP 隐层 |
| `--solar_power_map_dropout` | `0.0` | MLP dropout |
| `--solar_power_fusion` | `mapped` | `mapped` \| `direct` \| `average` \| `gated` \| `learned` |
| `--solar_power_direct_head` | `linear` | `enc_in=3` 时 fallback direct head：`linear` \| `mlp` |
| `--solar_power_direct_hidden` | `16` | fallback direct MLP 隐层 |
| `--solar_power_fusion_hidden` | `16` | learned fusion gate 隐层 |
| `--solar_power_fusion_gate_init` | `2.0` | 初始更偏向 mapped 分支的门控 logit |
| `--solar_power_gamma_pdc` | `-0.004` | PVWatts \(\gamma\) |
| `--solar_power_temp_ref` | `25` | 参考温度 |
| `--solar_power_temp_amb` | `20` | 模型内缺气温时的环境温度 |
| `--solar_power_noct` | `45` | NOCT |
| `--solar_power_temp_cell_method` | `noct` | 模型内 `noct` \| `faiman` |
| `--solar_power_faiman_u0/u1` | `25` / `6.84` | Faiman 系数 |
| `--solar_power_wind_default` | `1.0` | 模型内缺风速时的默认值 |

只监督功率时请设：

```text
--solar_power_output power
```

同时监督 `[ghi,dni,dhi,power]` 时请设：

```text
--solar_power_output append
```

（`power` 模式下 `run.py` 会将 `target` 置为 `power`。）

---

## 6. 推荐实验矩阵

| 目的 | 数据 | map | prior | fusion | capacity |
|------|------|-----|-------|--------|----------|
| 离线物理 prior 质量 | 电站 CSV | — | 看 `power_pvlib` | — | auto `pdc0` |
| 线性 prior + 残差 | 电站 / Congo | `residual_mlp` | `linear` | `gated` | 估好的 \(C\) |
| **主推：PVWatts + 残差 + 融合** | 电站 | `residual_mlp` | `pvwatts` | `learned` | builder 的 `pdc0` |
| 纯学习映射 + 融合 | 电站 | `learned` | 任意（不进加和） | `learned` | 尺度参考 |
| 严格 GHI-only 映射 + 融合 | 最小列数据 | `ghi_mlp` | 任意（不进加和） | `gated`/`learned` | 尺度参考 |
| 仅 prior 无学习 | 任意 | `fixed` | `linear`/`pvwatts` | `mapped` | 对齐标签尺度 |

---

## 7. 数据流与模型约束

1. **SolarPhysics 仍要求 `enc_in=3`**：`[ghi, dni, dhi]`。  
   电站无 DNI/DHI 时，离线用 Erbs 补上，仅服务物理投影；**功率标签仍是真实 `power`**。
2. 模型输入只取前 3 通道；`power` 只作监督目标。  
3. 模型内 `pvwatts` 当前用 CLI 默认 \(T_a,W_s\) 估 \(T_c\)；CSV 中的 `temp_air/wind_speed` 供分析，接口已预留后续接入。  
4. 物理投影源码 `layers/solar_physics.py` **不修改**；映射全部在投影之后。

---

## 8. 最小变量清单（Checklist）

**必须有**

- [ ] `dtime` / `timeStamp`
- [ ] `observe_ghi` → `ghi`
- [ ] `observe_power` → `power`（真实功率；Congo 则用合成）
- [ ] `latitude`, `longitude`（及合理 `timezone`）

**强烈建议**

- [ ] `TEMP`, `WS`（离线 Faiman / 标定）
- [ ] 估好的 `pdc0` → `--solar_power_capacity`

**可选**

- [ ] 预报 GHI / 云 / 降水等（预报任务）
- [ ] `govern_flag` 清洗
- [ ] Erbs `dni,dhi`（跑 SolarPhysics 三通道时）

---

## 9. 常见问题

**Q: 只有电站三列，没有天气？**  
可以。离线脚本不用 `--weather`；模型用 `--solar_power_prior pvwatts` + 默认 `temp_amb/wind_default`，或改用 `linear` prior。

**Q: `capacity` 设多少？**  
优先用 builder 打印的 auto `pdc0`。设成 `1.0` 而标签是 kW 量级会导致 prior 尺度错位，残差难学。

**Q: 和 Congo 三通道物理实验什么关系？**  
- Congo：主打 GHI–DNI–DHI 可行集投影。  
- 电站：主打 **真实功率**；DNI/DHI 仅为兼容物理头。  
两条线可共用 `SolarPowerMap`。

**Q: 短临 5min / 超短期 4h / 短期 1d？**  
- 映射头本身是「同时刻 GHI→P」。  
- 时序预测改 `pred_len`，预报特征用天气表 `interval*` / `report_*` 对齐（需在数据管线中按任务再切）。

---

## 10. 文件速查

```text
scripts/solar/
  README.md                      # 本文件
  build_plant_power_dataset.py   # 电站 + pvlib 离线列
  build_power_dataset.py         # Congo 合成 power
  plant_power.sh                 # 电站一键
  power.sh                       # Congo/合成功率实验
  irr.sh / phy.sh                # 辐照基线 / 物理辐照实验

models/_solar_physics_utils.py   # SolarPowerMap, pvwatts_dc, init...
layers/solar_physics.py          # 物理投影（不改）
data/solar/*.csv                 # 训练数据放置目录
```
