# Physics-Constrained Decoder (PCD): 三类任务设计文档（最终版）

## 1. 目标

本文档将 `Physics-Constrained Decoder (PCD)` 统一为一类“先预测，再映射到物理可行集”的时间序列预测方法，并落地到三个代表性任务：

1. `闭包 (closure)`：太阳辐照度预测
2. `守恒 (conservation)`：水库蓄水量预测
3. `容量上界 (capacity upper bound)`：风电功率预测

三个任务分别对应三类最典型、也最容易建立理论分析的物理结构：

- 等式闭包
- 动态守恒
- 不等式上界

它们的共同形式是

$$
\hat y_{t+1:t+H} = \Pi_{\mathcal F_t}\big(g_\theta(h_t)\big),
$$

其中：

- $h_t$ 是 backbone 输出的隐藏状态
- $g_\theta(h_t)$ 是 unconstrained decoder 的原始预测
- $\mathcal F_t$ 是由物理规律定义的时变可行集
- $\Pi_{\mathcal F_t}$ 是投影、闭式构造或约束解码算子

我们希望强调的核心主张是：

- `PCD` 输出天然物理可行
- 若真实标签本身可行，则投影不会比原始输出更差
- 若投影是 `1-Lipschitz` 或非扩张映射，则模型复杂度上界不会变得更坏

---

## 2. 统一写法

设未来预测目标为 $y^\star \in \mathbb R^m$，原始 decoder 输出为

$$
z = g_\theta(h_t).
$$

`PCD` 的输出定义为

$$
\hat y = D_{\text{phys}}(z; c_t),
$$

其中 $c_t$ 表示任务相关的外部条件，例如：

- solar 中的太阳天顶角 $z_t$
- reservoir 中的时间间隔 $\Delta t$
- wind 中的数据驱动容量上界 $P_{\text{cap}}$ 或动态上界 $U_t$

如果 $D_{\text{phys}}$ 可以写成对凸可行集 $\mathcal F_t$ 的欧氏投影

$$
D_{\text{phys}}(z; c_t) = \Pi_{\mathcal F_t}(z),
$$

则通常可以得到三类标准性质。

### 性质 1：物理可行性

$$
\hat y \in \mathcal F_t
$$

即输出一定满足物理约束。

### 性质 2：对可行标签的误差不增大

若 $y^\star \in \mathcal F_t$，则

$$
\|\Pi_{\mathcal F_t}(z) - y^\star\|_2
\le
\|z - y^\star\|_2.
$$

因此只要真实标签本身位于可行集内，投影至少不会比原始 decoder 更差。

### 性质 3：容量上界不劣化

若 $\Pi_{\mathcal F_t}$ 是非扩张映射，则

$$
\|\Pi_{\mathcal F_t}(z_1) - \Pi_{\mathcal F_t}(z_2)\|_2
\le
\|z_1 - z_2\|_2.
$$

因此在 Rademacher complexity、Lipschitz complexity 或 covering number 分析里，`PCD` 不会增加函数类复杂度，常常可以得到“不比原始 decoder 更松”的泛化上界。

---

## 3. 任务一：闭包约束 `Closure`

### 3.1 任务定义

预测未来太阳辐照度三元组

$$
y_t = (\mathrm{GHI}_t, \mathrm{DNI}_t, \mathrm{DHI}_t),
$$

其中：

- `GHI`：Global Horizontal Irradiance
- `DNI`：Direct Normal Irradiance
- `DHI`：Diffuse Horizontal Irradiance

### 3.2 物理约束

太阳辐照度满足经典闭包关系

$$
\mathrm{GHI}_t = \mathrm{DNI}_t \cos z_t + \mathrm{DHI}_t,
$$

其中 $z_t$ 是太阳天顶角。进一步考虑昼夜与非负性，可定义可行集

$$
\mathcal F_t^{\text{solar}}
=
\left\{
(g,n,d):
g = n \cos z_t + d,\;
g \ge 0,\;
n \ge 0,\;
d \ge 0
\right\}.
$$

若需要，还可加入弱上界

$$
g \le I_0 \alpha \max(\cos z_t, 0),
$$

其中 $I_0$ 为太阳常数，$\alpha$ 为可学习或可估计系数。

### 3.3 为什么适合 PCD

- 闭包关系精确、干净、低噪声
- 可直接写成解析解码器，而不只是 soft loss
- 很容易证明可行性、内部可表达性和近地平线稳定性

### 3.4 数据需求

必须包含：

- `timestamp`
- `GHI`
- `DNI`
- `DHI`
- `latitude/longitude` 或 `solar_zenith`

建议包含：

- `temperature`
- `pressure`
- `humidity`
- `wind_speed`
- `cloud_cover`

### 3.5 可写的理论点

- `Physical feasibility theorem`
- `Interior expressivity theorem`
- `Near-horizon stability theorem`

---

## 4. 任务二：守恒约束 `Conservation`

### 4.1 任务定义

水库任务的核心是预测未来的动态状态。最自然的基本版本不是只预测 `storage`，而是联合预测

$$
y_t = (S_t, I_t, O_t),
$$

其中：

- $S_t$：storage，库容或蓄水量
- $I_t$：inflow，入流
- $O_t$：outflow，出流

你当前已经具备：

- `storage`
- `inflow`
- `outflow`

此外，你还能补充：

- `evaporation`

记蒸发相关变量为 $E_t$。因此有两个推荐版本：

- 基础版：只做 $(S_t, I_t, O_t)$ 的守恒约束
- 增强版：加入 $E_t$，做 evaporation-aware conservation

### 4.2 核心物理约束

离散时间下，不考虑蒸发时的基本守恒可写为

$$
S_{t+1} = S_t + \Delta t (I_t - O_t).
$$

若显式考虑蒸发损失，则更推荐使用

$$
S_{t+1} = S_t + \Delta t (I_t - O_t - E_t),
$$

其中 $E_t \ge 0$ 表示蒸发引起的有效出库损失。若还考虑渗漏、未观测泄放等额外损失项，可进一步扩展为

$$
S_{t+1} = S_t + \Delta t (I_t - O_t - E_t - L_t), \qquad E_t \ge 0,\; L_t \ge 0.
$$

因此可行集可定义为

$$
\mathcal F_t^{\text{res}}
=
\left\{
(S_{t+1}, I_t, O_t, E_t, L_t):
S_{t+1} = S_t + \Delta t (I_t - O_t - E_t - L_t),\;
I_t \ge 0,\;
O_t \ge 0,\;
E_t \ge 0,\;
L_t \ge 0
\right\}.
$$

如果暂时不建模 $L_t$，可以先使用只含蒸发项的版本；从论文稳健性看，额外加入一个 nonnegative slack 仍然是很好的选择，因为它更容易解释测量噪声和遗漏项。

### 4.3 `evaporation` 能怎么用

若补充了蒸发数据 $E_t$，它至少有三种价值：

1. 作为守恒方程中的显式损失项，直接提升物理约束的真实性
2. 作为辅助预测目标，与 `storage`、`inflow`、`outflow` 一起形成更完整的系统状态
3. 作为外生输入，帮助模型识别季节性与天气驱动的储量变化

如果蒸发数据本身已经是“体积损失率”或与 `storage` 同量纲的时间损失项，那么可以直接写进守恒式。  
如果蒸发数据是深度、通量或其他单位，例如 `mm/day`，则更稳妥的写法是先做单位转换，得到有效损失项

$$
E_t^{\text{vol}} = \phi(\text{evaporation}_t),
$$

再把 $E_t^{\text{vol}}$ 写入守恒方程。若暂时无法精确完成量纲变换，则建议：

- 把 `evaporation` 作为输入特征
- 或把它作为辅助预测目标
- 守恒硬约束先保留在 $(S, I, O)$ 或 $(S, I, O, L)$ 层面

### 4.4 为什么不建议只预测 `storage`

如果模型只输出 $S_{t+1:t+H}$，但未来的 $I_t, O_t$ 既不是已知量、也不是联合预测变量，那么守恒约束无法在未来预测阶段被硬施加。  
因此更适合的 `PCD` 设计是：

- decoder 先预测原始的 $(\tilde S, \tilde I, \tilde O)$
- 再通过守恒关系映射到可行集合

如果使用蒸发增强版，则更推荐：

- decoder 先预测原始的 $(\tilde S, \tilde I, \tilde O, \tilde E)$
- 再通过 evaporation-aware conservation 映射到可行集合

如果未来蒸发是外部可得信号，例如来自气象预报，也可以把 $E_t$ 当作已知条件输入，而不是预测变量。

### 4.5 数据需求

当前核心字段：

- `timestamp`
- `storage`
- `inflow`
- `outflow`
- `evaporation`

最关键的补充信息不是更多变量，而是：

- 明确采样频率或时间间隔 $\Delta t$
- 明确单位，例如 `m^3`、`m^3/s`
- 明确 `evaporation` 的单位及其是否能换算成 storage loss

建议额外收集：

- `rainfall`
- `release/spill detail`

### 4.6 可写的理论点

- `Affine feasibility`
- `Projection onto conservation set`
- `Error non-increase for feasible targets`

若使用显式蒸发项并保持凸可行集，则理论仍然比较干净，而且比“全部损失都吸收到 slack 中”的版本更有物理解释性。

---

## 5. 任务三：容量上界 `Capacity Upper Bound`

### 5.1 任务定义

预测未来风电功率

$$
y_t = P_t.
$$

你当前能使用的变量严格限定为：

- `density`
- `power`
- `pressure`
- `temperature`
- `wind_direction`
- `wind_speed`

因此风电任务的设计必须完全建立在这六个变量之上，不依赖额外的机组铭牌信息、叶轮参数或 turbine metadata。

### 5.2 物理约束

最基本的不等式约束是

$$
0 \le P_t \le P_{\text{cap}},
$$

其中 $P_{\text{cap}}$ 不是外部提供的额定容量，而是从训练集估计得到的经验容量上界。一个可行定义是

$$
P_{\text{cap}} = Q_q\big(\{P_i^{\text{train}}\}\big), \qquad q \in [0.995, 1].
$$

也就是说，我们用训练集的高分位功率或训练集最大功率来近似该风场的可达容量。

若希望进一步体现资源驱动上界，可使用

$$
0 \le P_t \le U_t,
$$

其中

$$
U_t = \min\left(P_{\text{cap}}, \alpha \rho_t v_t^3\right).
$$

这里：

- $\rho_t$ 对应 `density`
- $v_t$ 对应 `wind_speed`
- $\alpha$ 是通过训练或校准吸收设备常数、单位换算和平均功率系数的标量

在这个版本中：

- `power` 是目标变量
- `density` 与 `wind_speed` 用于构造动态上界
- `pressure`、`temperature`、`wind_direction` 作为上下文特征输入 backbone

### 5.3 为什么这个设计适合你当前数据

这个版本有三个优点：

1. 不依赖你拿不到的 `rated capacity`
2. 仍然保留了明确的不等式可行集
3. 能把 `density` 和 `wind_speed` 直接写进物理上界，而不是只当普通特征

因此它非常适合写成“弱物理上界”或“数据驱动物理上界”的 `PCD` 示例。

### 5.4 对应的 PCD 形式

最简单的 `PCD` 可写成区间投影

$$
\hat P_t = \min\{\max(\tilde P_t, 0), U_t\}.
$$

若不使用动态上界，则退化为

$$
\hat P_t = \min\{\max(\tilde P_t, 0), P_{\text{cap}}\}.
$$

这两种形式都天然满足：

- 输出物理可行
- 对任意可行真值的平方误差不增大
- 投影是 `1-Lipschitz`

### 5.5 可写的理论点

- `Feasibility by interval projection`
- `Pointwise error non-increase`
- `Non-expansiveness and capacity control`

这是三类任务里最干净的不等式约束例子。

---

## 6. 三个任务如何统一成同一篇论文

最推荐的主标题方向是：

> `Physics-Constrained Decoder for Time-Series Forecasting`

副标题或正文主线可以写成：

> `A unified decoder-level framework for closure, conservation, and capacity constraints`

三个任务在论文中的角色分别是：

- `Solar`：展示精确闭包结构
- `Reservoir`：展示动态守恒关系
- `Wind`：展示不等式容量约束

这三者合起来，能支持一个比“仅 solar 方法”更一般的论点：

> `PCD` 不是某个特定物理系统的技巧，而是一类适用于多种物理可行集的统一解码范式。

---

## 7. 推荐的理论结构

若目标是往 NeurIPS 风格推进，最稳妥的理论主线是下面三条，而不是直接宣称“误差上界一定更低”。

### Theorem 1: Physical feasibility

对任意输入和任意时刻，`PCD` 输出满足对应任务定义的物理可行约束。

### Theorem 2: Error non-increase on feasible labels

若真实标签属于物理可行集，则相对于原始 decoder 输出，`PCD` 投影后的输出不会增加到真实标签的欧氏误差。

### Theorem 3: Capacity/generalization control

若 `PCD` 解码器是非扩张映射，则复合函数类

$$
\mathcal H_{\text{PCD}} = \Pi_{\mathcal F} \circ \mathcal H
$$

的复杂度不超过原始函数类 $\mathcal H$，因此标准泛化上界不会更差。

这条路线的优势是：

- 结论稳妥
- 与三类任务统一
- 审稿人更容易接受

---

## 8. 实验组织建议

### 8.1 对比配置

建议每个任务至少做以下对比：

- `Backbone`
- `Backbone + soft physics loss`
- `Backbone + PCD`

如有余力，再加入：

- `Backbone + PCD + auxiliary loss`

这样可以清楚地区分“仅做正则化”和“decoder-level hard feasibility”的效果差异。

### 8.2 评价指标

除常规预测误差外，建议加入物理一致性指标：

- `Closure violation`
- `Conservation residual`
- `Upper-bound violation rate`

如果 reservoir 使用了 `evaporation`，还可以增加：

- `Evaporation-aware conservation residual`

### 8.3 建议消融

- 是否使用投影
- reservoir 中是否加入 slack
- wind 中使用常数上界还是动态上界
- solar 中是否加入 weak upper bound
- reservoir 中是否显式使用 `evaporation`
- reservoir 中将 `evaporation` 作为输入、辅助目标还是硬约束项

---

## 9. 当前数据准备情况（按最终可执行版本）

### 9.1 Wind

当前可用变量严格为：

- `density`
- `power`
- `pressure`
- `temperature`
- `wind_direction`
- `wind_speed`

结论：

- 已经足够支撑 `capacity upper bound` 任务
- 不再依赖 `rated capacity`
- 容量上界由训练集估计的 $P_{\text{cap}}$ 给出
- 若进一步使用 $U_t = \min(P_{\text{cap}}, \alpha \rho_t v_t^3)$，则能形成更强的动态上界版本

### 9.2 Reservoir

当前可用变量为：

- `storage`
- `inflow`
- `outflow`
- `evaporation`

结论：

- 已经足够支撑基础版 `conservation` 任务
- 若时间间隔和单位明确，可以直接写守恒约束
- `evaporation` 可以显式进入损失项
- 若 `evaporation` 单位能对齐到 storage loss，可升级为更强的 evaporation-aware conservation 版本

### 9.3 Closure

若要凑齐三条主线，还需要补一个 solar 数据集，最关键字段是：

- `GHI`
- `DNI`
- `DHI`
- `solar_zenith` 或 `lat/lon + timestamp`

---

## 10. 最终建议

如果你想尽快形成一版论文骨架，最推荐的三任务组合仍然是：

1. `Solar irradiance forecasting` for `closure`
2. `Reservoir forecasting` for `conservation`
3. `Wind power forecasting` for `capacity upper bound`

其中：

- solar 负责“精确等式闭包”
- reservoir 负责“动态守恒”
- wind 负责“硬不等式上界”

结合你当前已经明确的数据条件，本文档对应的最终实现路线是：

- wind 只使用 `density, power, pressure, temperature, wind_direction, wind_speed`
- reservoir 使用 `storage, inflow, outflow`，并补充 `evaporation`
- wind 不再假设外部额定容量，改用训练集估计容量上界
- reservoir 先做守恒主线，再把 `evaporation` 作为显式损失项并入约束

这条路线比较稳，既贴合你手头真实可得的数据，也保留了足够强的理论和论文叙事空间。
