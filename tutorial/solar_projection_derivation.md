# Solar Physics Projection Derivation

本文推导 `layers/solar_physics.py` 中 `project_to_feasible_set()` 的核心公式，说明白天条件下如何将原始预测点

$$
x_0 = (g, n, d)
$$

投影到太阳物理可行集合

$$
\mathcal{C}(c) = \{(G, N, D): G = cN + D,\; N \ge 0,\; D \ge 0\}
$$

其中：

- $G$ 表示 `ghi`
- $N$ 表示 `dni`
- $D$ 表示 `dhi`
- $c = \cos(\text{zenith})$

夜间情形 $c_{raw} \le \text{night\_cos\_threshold}$ 时，代码直接返回原点 $(0, 0, 0)$，因此下面的推导只讨论白天投影。

## 1. 优化问题

给定原始点 $(g, n, d)$，希望找到离它最近的物理合法点 $(G, N, D)$。对应的优化问题是

$$
\min_{G,N,D} \frac12 \left[(G-g)^2 + (N-n)^2 + (D-d)^2\right]
$$

subject to

$$
G - cN - D = 0,\quad N \ge 0,\quad D \ge 0
$$

这里加上 $1/2$ 只是为了让导数更简洁，不影响最优解。

## 2. 内部解：先忽略非负约束

先暂时忽略

$$
N \ge 0,\quad D \ge 0
$$

只保留等式约束

$$
G - cN - D = 0
$$

这时拉格朗日函数为

$$
\mathcal{L}(G,N,D,\lambda)
=
\frac12 \left[(G-g)^2 + (N-n)^2 + (D-d)^2\right]
+ \lambda(G - cN - D)
$$

### 2.1 对各变量求偏导

对 $G$ 求偏导：

$$
\frac{\partial \mathcal{L}}{\partial G} = G - g + \lambda = 0
$$

因此

$$
G = g - \lambda
$$

对 $N$ 求偏导：

$$
\frac{\partial \mathcal{L}}{\partial N} = N - n - c\lambda = 0
$$

因此

$$
N = n + c\lambda
$$

对 $D$ 求偏导：

$$
\frac{\partial \mathcal{L}}{\partial D} = D - d - \lambda = 0
$$

因此

$$
D = d + \lambda
$$

对 $\lambda$ 求偏导得到原约束：

$$
\frac{\partial \mathcal{L}}{\partial \lambda} = G - cN - D = 0
$$

### 2.2 先求 $\lambda$

把

$$
G = g - \lambda,\quad N = n + c\lambda,\quad D = d + \lambda
$$

代入约束

$$
G - cN - D = 0
$$

可得

$$
(g - \lambda) - c(n + c\lambda) - (d + \lambda) = 0
$$

展开：

$$
g - \lambda - cn - c^2\lambda - d - \lambda = 0
$$

整理得到：

$$
g - cn - d - (c^2 + 2)\lambda = 0
$$

因此

$$
\lambda = \frac{g - cn - d}{c^2 + 2}
$$

### 2.3 求 $n_{int}$

由

$$
N = n + c\lambda
$$

代入上式：

$$
N = n + c \cdot \frac{g - cn - d}{c^2 + 2}
$$

通分：

$$
N = \frac{n(c^2 + 2) + cg - c^2n - cd}{c^2 + 2}
$$

化简：

$$
N = \frac{cg + 2n - cd}{c^2 + 2}
$$

因此内部解中的 $N$ 为

$$
n_{int} = \frac{cg + 2n - cd}{c^2 + 2}
$$

这正是代码中的

```python
n_int = (c * g + 2.0 * n - c * d) / (c.square() + 2.0)
```

### 2.4 求 $d_{int}$

由

$$
D = d + \lambda
$$

代入 $\lambda$：

$$
D = d + \frac{g - cn - d}{c^2 + 2}
$$

通分：

$$
D = \frac{d(c^2 + 2) + g - cn - d}{c^2 + 2}
$$

化简：

$$
D = \frac{g + (c^2 + 1)d - cn}{c^2 + 2}
$$

因此内部解中的 $D$ 为

$$
d_{int} = \frac{g + (c^2 + 1)d - cn}{c^2 + 2}
$$

这正是代码中的

```python
d_int = (g + (c.square() + 1.0) * d - c * n) / (c.square() + 2.0)
```

### 2.5 求 $g_{int}$

内部解必须满足平面约束

$$
G = cN + D
$$

所以

$$
g_{int} = c\,n_{int} + d_{int}
$$

对应代码

```python
g_int = c * n_int + d_int
```

### 2.6 内部解何时有效

上面求得的是“投影到平面 $G = cN + D$ 上的最近点”，但真正的可行集合还要求

$$
N \ge 0,\quad D \ge 0
$$

因此只有在

$$
n_{int} \ge 0,\quad d_{int} \ge 0
$$

时，这个内部解才是真正可行的。

代码中用

```python
interior_valid = (n_int >= 0.0) & (d_int >= 0.0)
```

来判断这一点。

## 3. 边界解一：$D = 0$

如果内部解不满足 $D \ge 0$ 或 $N \ge 0$，最优点可能落在边界上。第一类边界是

$$
D = 0
$$

这时约束变为

$$
G = cN,\quad N \ge 0
$$

所以只需对单变量 $N$ 做优化：

$$
\min_{N \ge 0} \frac12 \left[(cN-g)^2 + (N-n)^2 + (0-d)^2\right]
$$

先忽略 $N \ge 0$，对 $N$ 求导：

$$
\frac{d}{dN}\frac12 \left[(cN-g)^2 + (N-n)^2 + d^2\right] = 0
$$

得到

$$
c(cN-g) + (N-n) = 0
$$

展开整理：

$$
c^2N - cg + N - n = 0
$$

$$
(c^2 + 1)N = cg + n
$$

因此无约束极值点为

$$
N = \frac{cg + n}{c^2 + 1}
$$

考虑非负约束后：

$$
N_{b1} = \max\left(0,\frac{cg+n}{c^2+1}\right)
$$

于是边界候选点为

$$
(G,N,D) = (cN_{b1},\; N_{b1},\; 0)
$$

这正是代码中的

```python
n_b1 = torch.clamp((c * g + n) / (c.square() + 1.0), min=0.0)
boundary_d_zero = torch.stack([c * n_b1, n_b1, zero], dim=-1)
```

## 4. 边界解二：$N = 0$

第二类边界是

$$
N = 0
$$

这时约束变为

$$
G = D,\quad D \ge 0
$$

优化问题变成

$$
\min_{D \ge 0} \frac12 \left[(D-g)^2 + (0-n)^2 + (D-d)^2\right]
$$

先忽略 $D \ge 0$，对 $D$ 求导：

$$
(D-g) + (D-d) = 0
$$

即

$$
2D - (g+d) = 0
$$

因此无约束极值点为

$$
D = \frac{g+d}{2}
$$

考虑非负约束后：

$$
D_{b2} = \max\left(0,\frac{g+d}{2}\right)
$$

于是边界候选点为

$$
(G,N,D) = (D_{b2},\; 0,\; D_{b2})
$$

对应代码

```python
d_b2 = torch.clamp((g + d) / 2.0, min=0.0)
boundary_n_zero = torch.stack([d_b2, zero, d_b2], dim=-1)
```

## 5. 角点：原点 $(0, 0, 0)$

当两条边界同时激活时：

$$
N = 0,\quad D = 0
$$

再结合

$$
G = cN + D
$$

得到

$$
G = 0
$$

因此角点就是

$$
(0,0,0)
$$

代码里将它作为一个独立候选：

```python
origin = torch.stack([zero, zero, zero], dim=-1)
```

## 6. 为什么只需要比较这四类候选

白天可行域

$$
\mathcal{C}(c) = \{(G, N, D): G = cN + D,\; N \ge 0,\; D \ge 0\}
$$

是一个凸集，可以理解为平面 $G = cN + D$ 上的一个二维楔形区域。对于欧氏投影问题，最优点只可能出现在：

- 内部
- 边界 $D = 0$
- 边界 $N = 0$
- 两条边界交点 $(0,0,0)$

因此代码只需要构造这四个候选点，分别计算它们到原始点 $(g,n,d)$ 的平方距离，然后选最小者即可。

## 7. 对应到代码的最终流程

代码实现顺序可以概括为：

1. 构造内部候选 `interior`
2. 判断其是否满足 $n_{int} \ge 0$ 且 $d_{int} \ge 0$
3. 构造边界候选 `boundary_d_zero`
4. 构造边界候选 `boundary_n_zero`
5. 构造角点 `origin`
6. 比较这四类候选的平方距离
7. 选择距离原始点最近的合法候选
8. 如果 $c_{raw} \le \text{night\_cos\_threshold}$，则直接返回 `origin`

## 8. 关键公式汇总

内部解：

$$
n_{int} = \frac{cg + 2n - cd}{c^2 + 2}
$$

$$
d_{int} = \frac{g + (c^2 + 1)d - cn}{c^2 + 2}
$$

$$
g_{int} = c\,n_{int} + d_{int}
$$

边界 $D = 0$：

$$
N_{b1} = \max\left(0,\frac{cg+n}{c^2+1}\right)
$$

$$
(G,N,D) = (cN_{b1}, N_{b1}, 0)
$$

边界 $N = 0$：

$$
D_{b2} = \max\left(0,\frac{g+d}{2}\right)
$$

$$
(G,N,D) = (D_{b2}, 0, D_{b2})
$$

夜间：

$$
c_{raw} \le \tau \Rightarrow (G,N,D) = (0,0,0)
$$

## 9. 与代码文件的对应关系

- 投影函数实现：`layers/solar_physics.py`
- 函数名：`project_to_feasible_set(raw_state, cos_zenith, night_cos_threshold=0.0)`

如果后续需要，也可以在这份文档基础上继续补一版 KKT 条件的统一推导，把内部解、边界解和角点放到同一套约束优化框架下说明。
