# Solar Physics Module NRMSE Comparison

This document compares three settings across solar datasets:

- **New physics module** (`result_new__Solar_*.txt` in this repo)
- **Old physics module** (`result_v2__Solar_*.txt` in this repo)
- **Baseline** (`/opt/data/private/code/Time-Series-Library-fm/result_new__Solar_*.txt`)

Metric: **NRMSE** (lower is better).

For Congo/Germany/NewDelhi:
- each value is the average over `pl = {24, 48, 96, 192}`.

For Folsom:
- new/old values are averaged over `pl = {24, 48, 96, 192}`;
- baseline uses `result_new__Solar_Folsom_combined_no_zero_False.txt`, where currently only one usable horizon exists per method (effectively `pl192`).

## Congo (average over 4 horizons)

| Method | New | Old | Baseline |
|---|---:|---:|---:|
| iTransformer | 0.115330 | 0.117155 | 0.122602 |
| PatchTST | 0.112133 | 0.117003 | 0.269981 |
| DLinear | 0.111884 | 0.117097 | 0.121792 |
| Crossformer | 0.114338 | 0.116940 | 0.247030 |
| Autoformer | 0.114043 | 0.117045 | 0.157574 |
| Informer | 0.116907 | 0.116841 | 0.216653 |
| FEDformer | 0.116054 | 0.116870 | 0.179442 |
| MICN | 0.179973 | 0.111566 | 0.135737 |
| LightTS | 0.109774 | 0.117076 | 0.113636 |
| PAttn | 0.111859 | 0.125561 | 0.114355 |
| TSMixer | 0.109481 | 0.116567 | 0.109194 |

## Germany (average over 4 horizons)

| Method | New | Old | Baseline |
|---|---:|---:|---:|
| iTransformer | 0.096758 | 0.103058 | 0.114106 |
| PatchTST | 0.112465 | 0.097008 | 0.240322 |
| DLinear | 0.092229 | 0.104551 | 0.100330 |
| Crossformer | 0.096915 | 0.105444 | 0.183580 |
| Autoformer | 0.287659 | 0.144868 | 0.145083 |
| Informer | 0.100014 | 0.104965 | 0.159059 |
| FEDformer | 0.103425 | 0.105010 | 0.197278 |
| MICN | 0.139456 | 0.099167 | 0.117423 |
| LightTS | 0.090069 | 0.097052 | 0.096022 |
| PAttn | 0.095210 | 0.097188 | 0.099897 |
| TSMixer | 0.089943 | 0.102967 | 0.096780 |

## NewDelhi (average over 4 horizons)

| Method | New | Old | Baseline |
|---|---:|---:|---:|
| iTransformer | 0.081014 | 0.096226 | 0.095603 |
| PatchTST | 0.082039 | 0.083879 | 0.232399 |
| DLinear | 0.075936 | 0.082649 | 0.079721 |
| Crossformer | 0.093011 | 0.100306 | 0.231382 |
| Autoformer | 0.085921 | 0.094401 | 0.166475 |
| Informer | 0.079436 | 0.097832 | 0.209031 |
| FEDformer | 0.094900 | 0.095143 | 0.154160 |
| MICN | 0.141122 | 0.088640 | 0.082878 |
| LightTS | 0.075748 | 0.084658 | 0.077951 |
| PAttn | 0.077927 | 0.089419 | 0.082314 |
| TSMixer | 0.075759 | 0.094129 | 0.077218 |

## Folsom_combined (new/old average over 4 horizons, baseline from no_zero available horizon)

| Method | New (4 horizons) | Old (4 horizons) | Baseline (no_zero) | Baseline horizons |
|---|---:|---:|---:|---:|
| iTransformer | 0.031917 | 0.046717 | 0.151894 | 1 |
| PatchTST | 0.028058 | 0.044699 | 0.177387 | 1 |
| DLinear | 0.027658 | 0.047738 | 0.145832 | 1 |
| Crossformer | 0.032275 | 0.048699 | 0.192992 | 1 |
| Autoformer | 0.050595 | 0.046409 | 0.085445 | 1 |
| Informer | 0.039550 | 0.049875 | 0.189437 | 1 |
| FEDformer | 0.045705 | 0.048986 | 0.080717 | 1 |
| MICN | 0.030935 | 0.043679 | 0.065739 | 1 |
| LightTS | 0.029195 | 0.043695 | 0.144804 | 1 |
| PAttn | 0.030733 | 0.044226 | 0.148428 | 1 |
| TSMixer | 0.036122 | 0.050355 | 0.137840 | 1 |

## Notes

- NRMSE values are extracted directly from result files.
- If duplicate runs exist for the same method and horizon, the minimum NRMSE is used for that horizon before averaging.
- Folsom baseline is currently not fully symmetric with other datasets because only one usable horizon is available in the provided `no_zero` file.
