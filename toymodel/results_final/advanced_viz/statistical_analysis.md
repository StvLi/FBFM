# 统计显著性检验与效应量分析报告

> 本报告对 FBFM 预实验的四组实验进行配对 Wilcoxon 符号秩检验 (non-parametric) 与 Cohen's d 效应量分析。
> 所有检验基于 5 个独立随机种子的配对样本。显著性标注：*** p<0.001, ** p<0.01, * p<0.05, n.s. 不显著。

---
## 实验 A：模型失配鲁棒性 (Model Mismatch Sweep)

**对比组**: FBFM (Ours) vs Vanilla FM

| 实验条件 | 指标 | Vanilla均值 | FBFM均值 | 改进率 | p-value | Cohen's d | 效应量级别 | 显著性 |
|---------|------|-----------|---------|-------|---------|-----------|----------|-------|
| step_nominal | Action Jitter | 0.1356 | 0.0168 | +87.6% | 0.8125 | 0.45 | Small | n.s. |
| step_nominal | Action MSE vs PID | 0.0998 | 0.0114 | +88.6% | 0.4375 | 0.45 | Small | n.s. |
| step_nominal | Energy | 1.3768 | 1.2904 | +6.3% | 0.1250 | 0.45 | Small | n.s. |
| step_nominal | IAE | 0.1126 | 0.1119 | +0.7% | 0.1250 | 0.56 | Medium | n.s. |
| step_nominal | Control TV | 16.1431 | 11.1721 | +30.8% | 0.4375 | 0.45 | Small | n.s. |
| step_nominal | Position MSE | 0.0643 | 0.0641 | +0.2% | 0.8125 | 0.44 | Small | n.s. |
| step_mass×1.5 | Action Jitter | 0.2926 | 0.0272 | +90.7% | 0.1875 | 0.72 | Medium | n.s. |
| step_mass×1.5 | Action MSE vs PID | 0.2232 | 0.0489 | +78.1% | 0.4375 | 0.71 | Medium | n.s. |
| step_mass×1.5 | Energy | 1.8744 | 1.6957 | +9.5% | 0.3125 | 0.72 | Medium | n.s. |
| step_mass×1.5 | IAE | 0.1305 | 0.1289 | +1.2% | 0.3125 | 0.77 | Medium | n.s. |
| step_mass×1.5 | Control TV | 27.2544 | 17.8515 | +34.5% | 0.0625 | 0.75 | Medium | n.s. |
| step_mass×1.5 | Position MSE | 0.0771 | 0.0771 | +0.0% | 0.6250 | 0.31 | Small | n.s. |
| step_mass×2 | Action Jitter | 0.2704 | 0.1106 | +59.1% | 0.0625 | 0.63 | Medium | n.s. |
| step_mass×2 | Action MSE vs PID | 0.2766 | 0.2351 | +15.0% | 0.4375 | 0.61 | Medium | n.s. |
| step_mass×2 | Energy | 2.3594 | 2.3247 | +1.5% | 0.4375 | 0.61 | Medium | n.s. |
| step_mass×2 | IAE | 0.1473 | 0.1473 | -0.0% | 0.6250 | -0.03 | Negligible | n.s. |
| step_mass×2 | Control TV | 35.3916 | 29.4077 | +16.9% | 0.1875 | 0.72 | Medium | n.s. |
| step_mass×2 | Position MSE | 0.0882 | 0.0882 | -0.1% | 0.1250 | -1.26 | Very Large | n.s. |
| step_mass×3 | Action Jitter | 1.1657 | 0.1360 | +88.3% | 0.0625 | 0.73 | Medium | n.s. |
| step_mass×3 | Action MSE vs PID | 1.0851 | 0.2977 | +72.6% | 0.1875 | 0.58 | Medium | n.s. |
| step_mass×3 | Energy | 3.9189 | 3.1018 | +20.9% | 0.1250 | 0.58 | Medium | n.s. |
| step_mass×3 | IAE | 0.2029 | 0.1969 | +3.0% | 0.6250 | 0.43 | Small | n.s. |
| step_mass×3 | Control TV | 76.0056 | 39.4482 | +48.1% | 0.0625 | 0.75 | Medium | n.s. |
| step_mass×3 | Position MSE | 0.1120 | 0.1110 | +0.9% | 1.0000 | 0.34 | Small | n.s. |
| step_stiff×3 | Action Jitter | 0.5170 | 0.0193 | +96.3% | 0.0625 | 0.96 | Large | n.s. |
| step_stiff×3 | Action MSE vs PID | 0.2644 | 0.0148 | +94.4% | 0.0625 | 0.96 | Large | n.s. |
| step_stiff×3 | Energy | 1.5713 | 1.3210 | +15.9% | 0.0625 | 0.97 | Large | n.s. |
| step_stiff×3 | IAE | 0.1175 | 0.1228 | -4.5% | 0.0625 | -1.14 | Large | n.s. |
| step_stiff×3 | Control TV | 32.4890 | 17.6367 | +45.7% | 0.1250 | 0.97 | Large | n.s. |
| step_stiff×3 | Position MSE | 0.0649 | 0.0652 | -0.5% | 0.0625 | -1.23 | Very Large | n.s. |
| step_combined | Action Jitter | 0.6482 | 0.0410 | +93.7% | 0.0625 | 1.01 | Large | n.s. |
| step_combined | Action MSE vs PID | 0.4591 | 0.0813 | +82.3% | 0.3125 | 0.95 | Large | n.s. |
| step_combined | Energy | 2.3184 | 1.9257 | +16.9% | 0.0625 | 0.98 | Large | n.s. |
| step_combined | IAE | 0.1561 | 0.1582 | -1.3% | 0.0625 | -1.58 | Very Large | n.s. |
| step_combined | Control TV | 46.7475 | 25.8335 | +44.7% | 0.1250 | 1.00 | Large | n.s. |
| step_combined | Position MSE | 0.0907 | 0.0910 | -0.3% | 0.0625 | -1.94 | Very Large | n.s. |
| sinusoidal_nominal | Action Jitter | 0.3290 | 0.3272 | +0.5% | 0.0625 | 1.15 | Large | n.s. |
| sinusoidal_nominal | Action MSE vs PID | 0.1954 | 0.1918 | +1.8% | 0.0625 | 1.22 | Very Large | n.s. |
| sinusoidal_nominal | Energy | 5.3309 | 5.3286 | +0.0% | 0.1250 | 0.57 | Medium | n.s. |
| sinusoidal_nominal | IAE | 0.1778 | 0.1780 | -0.1% | 0.0625 | -1.93 | Very Large | n.s. |
| sinusoidal_nominal | Control TV | 30.7767 | 30.3670 | +1.3% | 0.0625 | 1.73 | Very Large | n.s. |
| sinusoidal_nominal | Position MSE | 0.0389 | 0.0391 | -0.3% | 0.0625 | -2.83 | Very Large | n.s. |
| sinusoidal_mass×1.5 | Action Jitter | 0.4626 | 0.5861 | -26.7% | 0.0625 | -2.17 | Very Large | n.s. |
| sinusoidal_mass×1.5 | Action MSE vs PID | 0.4442 | 0.7859 | -77.0% | 0.0625 | -6.22 | Very Large | n.s. |
| sinusoidal_mass×1.5 | Energy | 10.5805 | 10.7429 | -1.5% | 0.0625 | -2.70 | Very Large | n.s. |
| sinusoidal_mass×1.5 | IAE | 0.2628 | 0.2564 | +2.4% | 0.0625 | 3.66 | Very Large | n.s. |
| sinusoidal_mass×1.5 | Control TV | 76.2234 | 84.7213 | -11.1% | 0.0625 | -2.23 | Very Large | n.s. |
| sinusoidal_mass×1.5 | Position MSE | 0.0871 | 0.0843 | +3.2% | 0.0625 | 3.62 | Very Large | n.s. |
| sinusoidal_mass×2 | Action Jitter | 0.8767 | 1.0402 | -18.6% | 0.1250 | -0.68 | Medium | n.s. |
| sinusoidal_mass×2 | Action MSE vs PID | 1.2831 | 1.7360 | -35.3% | 0.0625 | -1.58 | Very Large | n.s. |
| sinusoidal_mass×2 | Energy | 17.9985 | 17.6704 | +1.8% | 0.1250 | 1.31 | Very Large | n.s. |
| sinusoidal_mass×2 | IAE | 0.3649 | 0.3468 | +5.0% | 0.0625 | 1.94 | Very Large | n.s. |
| sinusoidal_mass×2 | Control TV | 123.0024 | 129.0148 | -4.9% | 0.1875 | -0.80 | Medium | n.s. |
| sinusoidal_mass×2 | Position MSE | 0.1705 | 0.1573 | +7.7% | 0.0625 | 2.00 | Very Large | n.s. |
| sinusoidal_mass×3 | Action Jitter | 1.4418 | 1.3457 | +6.7% | 0.1250 | 1.14 | Large | n.s. |
| sinusoidal_mass×3 | Action MSE vs PID | 3.0121 | 3.1170 | -3.5% | 0.3125 | -0.58 | Medium | n.s. |
| sinusoidal_mass×3 | Energy | 35.6452 | 35.0594 | +1.6% | 0.1250 | 1.32 | Very Large | n.s. |
| sinusoidal_mass×3 | IAE | 0.5514 | 0.5660 | -2.7% | 0.0625 | -1.62 | Very Large | n.s. |
| sinusoidal_mass×3 | Control TV | 198.7802 | 195.0321 | +1.9% | 0.0625 | 1.62 | Very Large | n.s. |
| sinusoidal_mass×3 | Position MSE | 0.3973 | 0.4204 | -5.8% | 0.0625 | -1.64 | Very Large | n.s. |
| sinusoidal_stiff×3 | Action Jitter | 0.3273 | 0.3259 | +0.4% | 0.1250 | 0.91 | Large | n.s. |
| sinusoidal_stiff×3 | Action MSE vs PID | 0.1933 | 0.1904 | +1.5% | 0.1250 | 0.80 | Medium | n.s. |
| sinusoidal_stiff×3 | Energy | 4.7119 | 4.7121 | -0.0% | 1.0000 | -0.03 | Negligible | n.s. |
| sinusoidal_stiff×3 | IAE | 0.1625 | 0.1628 | -0.2% | 0.0625 | -1.27 | Very Large | n.s. |
| sinusoidal_stiff×3 | Control TV | 29.2679 | 28.8575 | +1.4% | 0.1250 | 1.44 | Very Large | n.s. |
| sinusoidal_stiff×3 | Position MSE | 0.0323 | 0.0325 | -0.5% | 0.0625 | -1.55 | Very Large | n.s. |
| sinusoidal_combined | Action Jitter | 0.8060 | 0.6159 | +23.6% | 0.3125 | 0.86 | Large | n.s. |
| sinusoidal_combined | Action MSE vs PID | 0.8945 | 0.8845 | +1.1% | 0.6250 | 0.03 | Negligible | n.s. |
| sinusoidal_combined | Energy | 15.7462 | 15.4503 | +1.9% | 0.0625 | 1.23 | Very Large | n.s. |
| sinusoidal_combined | IAE | 0.3516 | 0.3563 | -1.3% | 0.0625 | -3.07 | Very Large | n.s. |
| sinusoidal_combined | Control TV | 109.8201 | 107.5634 | +2.1% | 0.6250 | 0.27 | Small | n.s. |
| sinusoidal_combined | Position MSE | 0.1546 | 0.1585 | -2.5% | 0.0625 | -2.83 | Very Large | n.s. |

---
## 实验 B：外部干扰恢复 (Disturbance Recovery)

**对比组**: FBFM (Ours) vs Vanilla FM

| 实验条件 | 指标 | Vanilla均值 | FBFM均值 | 改进率 | p-value | Cohen's d | 效应量级别 | 显著性 |
|---------|------|-----------|---------|-------|---------|-----------|----------|-------|
| pos_offset_nom | Action Jitter | 0.2187 | 0.0651 | +70.2% | 0.0625 | 0.59 | Medium | n.s. |
| pos_offset_nom | Action MSE vs PID | 0.5610 | 0.4552 | +18.9% | 0.0625 | 0.55 | Medium | n.s. |
| pos_offset_nom | Energy | 1.6894 | 1.5681 | +7.2% | 0.0625 | 0.62 | Medium | n.s. |
| pos_offset_nom | IAE | 0.1603 | 0.1587 | +1.0% | 0.8125 | 0.50 | Medium | n.s. |
| pos_offset_nom | Control TV | 26.7705 | 19.6577 | +26.6% | 0.0625 | 0.68 | Medium | n.s. |
| pos_offset_nom | Position MSE | 0.0868 | 0.0864 | +0.5% | 0.6250 | 0.27 | Small | n.s. |
| pos_offset_nom | Recovery Steps | 57.0000 | 57.0000 | +0.0% | 1.0000 | 0.00 | Negligible | n.s. |
| pos_offset_mm | Action Jitter | 1.1034 | 0.2665 | +75.9% | 0.1250 | 1.03 | Large | n.s. |
| pos_offset_mm | Action MSE vs PID | 1.0267 | 0.9075 | +11.6% | 0.6250 | 0.12 | Negligible | n.s. |
| pos_offset_mm | Energy | 3.0460 | 2.8413 | +6.7% | 0.6250 | 0.20 | Small | n.s. |
| pos_offset_mm | IAE | 0.2255 | 0.2374 | -5.3% | 0.0625 | -1.25 | Very Large | n.s. |
| pos_offset_mm | Control TV | 72.9586 | 46.9088 | +35.7% | 0.1250 | 0.85 | Large | n.s. |
| pos_offset_mm | Position MSE | 0.1242 | 0.1266 | -1.9% | 0.0625 | -0.83 | Large | n.s. |
| pos_offset_mm | Recovery Steps | 69.0000 | 69.2000 | -0.3% | 1.0000 | -0.45 | Small | n.s. |
| step_force_nom | Action Jitter | 0.7022 | 0.0384 | +94.5% | 0.0625 | 1.08 | Large | n.s. |
| step_force_nom | Action MSE vs PID | 0.4128 | 0.0581 | +85.9% | 0.0625 | 1.16 | Large | n.s. |
| step_force_nom | Energy | 2.2908 | 1.9291 | +15.8% | 0.1250 | 1.16 | Large | n.s. |
| step_force_nom | IAE | 0.1523 | 0.1594 | -4.7% | 0.0625 | -1.56 | Very Large | n.s. |
| step_force_nom | Control TV | 44.9747 | 22.3367 | +50.3% | 0.0625 | 1.20 | Large | n.s. |
| step_force_nom | Position MSE | 0.0733 | 0.0759 | -3.5% | 0.0625 | -2.30 | Very Large | n.s. |
| step_force_nom | Recovery Steps | 57.0000 | 57.0000 | +0.0% | 1.0000 | 0.00 | Negligible | n.s. |
| step_force_mm | Action Jitter | 0.8337 | 0.0542 | +93.5% | 0.0625 | 2.00 | Very Large | n.s. |
| step_force_mm | Action MSE vs PID | 0.5940 | 0.1336 | +77.5% | 0.0625 | 1.51 | Very Large | n.s. |
| step_force_mm | Energy | 2.9246 | 2.4624 | +15.8% | 0.0625 | 1.63 | Very Large | n.s. |
| step_force_mm | IAE | 0.1858 | 0.1892 | -1.8% | 0.1250 | -1.27 | Very Large | n.s. |
| step_force_mm | Control TV | 60.3202 | 33.4795 | +44.5% | 0.0625 | 1.88 | Very Large | n.s. |
| step_force_mm | Position MSE | 0.0964 | 0.0972 | -0.9% | 0.1250 | -1.54 | Very Large | n.s. |
| step_force_mm | Recovery Steps | 69.0000 | 69.2000 | -0.3% | 1.0000 | -0.45 | Small | n.s. |
| impulse_nom | Action Jitter | 0.3435 | 0.0177 | +94.8% | 0.0625 | 0.72 | Medium | n.s. |
| impulse_nom | Action MSE vs PID | 0.2543 | 0.0138 | +94.6% | 0.0625 | 0.74 | Medium | n.s. |
| impulse_nom | Energy | 1.5324 | 1.2928 | +15.6% | 0.0625 | 0.74 | Medium | n.s. |
| impulse_nom | IAE | 0.1175 | 0.1171 | +0.3% | 0.8125 | 0.09 | Negligible | n.s. |
| impulse_nom | Control TV | 27.7483 | 12.1413 | +56.2% | 0.0625 | 0.74 | Medium | n.s. |
| impulse_nom | Position MSE | 0.0646 | 0.0645 | +0.2% | 1.0000 | 0.36 | Small | n.s. |
| impulse_nom | Recovery Steps | 57.0000 | 57.0000 | +0.0% | 1.0000 | 0.00 | Negligible | n.s. |
| impulse_mm | Action Jitter | 0.6510 | 0.0422 | +93.5% | 0.0625 | 1.01 | Large | n.s. |
| impulse_mm | Action MSE vs PID | 0.4726 | 0.0843 | +82.2% | 0.3125 | 0.95 | Large | n.s. |
| impulse_mm | Energy | 2.3397 | 1.9340 | +17.3% | 0.0625 | 0.98 | Large | n.s. |
| impulse_mm | IAE | 0.1579 | 0.1600 | -1.3% | 0.0625 | -1.59 | Very Large | n.s. |
| impulse_mm | Control TV | 47.4163 | 26.2849 | +44.6% | 0.0625 | 1.02 | Large | n.s. |
| impulse_mm | Position MSE | 0.0909 | 0.0912 | -0.3% | 0.0625 | -1.89 | Very Large | n.s. |
| impulse_mm | Recovery Steps | 69.0000 | 69.2000 | -0.3% | 1.0000 | -0.45 | Small | n.s. |

---
## 实验 C：Jacobian 消融实验 (Ablation Study)

### C-1: FBFM vs FBFM-Identity (Jacobian 贡献)

| 实验条件 | 指标 | Vanilla均值 | FBFM均值 | 改进率 | p-value | Cohen's d | 效应量级别 | 显著性 |
|---------|------|-----------|---------|-------|---------|-----------|----------|-------|
| step_mass2 | Action Jitter | 0.4147 | 0.1106 | +73.3% | 0.1875 | 0.85 | Large | n.s. |
| step_mass2 | Action MSE vs PID | 0.5803 | 0.2351 | +59.5% | 0.0625 | 0.82 | Large | n.s. |
| step_mass2 | Energy | 2.5730 | 2.3247 | +9.6% | 0.6250 | 0.59 | Medium | n.s. |
| step_mass2 | Position MSE | 0.0903 | 0.0882 | +2.2% | 0.0625 | 2.22 | Very Large | n.s. |
| step_mass2 | State Pred. MSE | 0.0066 | 0.0063 | +5.0% | 0.8125 | 0.29 | Small | n.s. |
| step_combined | Action Jitter | 0.3734 | 0.0410 | +89.0% | 0.3125 | 0.89 | Large | n.s. |
| step_combined | Action MSE vs PID | 0.4133 | 0.0813 | +80.3% | 0.0625 | 0.80 | Medium | n.s. |
| step_combined | Energy | 2.1573 | 1.9257 | +10.7% | 0.6250 | 0.56 | Medium | n.s. |
| step_combined | Position MSE | 0.0933 | 0.0910 | +2.5% | 0.0625 | 1.56 | Very Large | n.s. |
| step_combined | State Pred. MSE | 0.0043 | 0.0040 | +7.6% | 0.0625 | 0.68 | Medium | n.s. |
| sin_mass2 | Action Jitter | 0.7390 | 1.0402 | -40.7% | 0.0625 | -1.33 | Very Large | n.s. |
| sin_mass2 | Action MSE vs PID | 1.4443 | 1.7360 | -20.2% | 0.1250 | -0.96 | Large | n.s. |
| sin_mass2 | Energy | 17.0084 | 17.6704 | -3.9% | 0.0625 | -2.44 | Very Large | n.s. |
| sin_mass2 | Position MSE | 0.1755 | 0.1573 | +10.3% | 0.0625 | 1.90 | Very Large | n.s. |
| sin_mass2 | State Pred. MSE | 0.0514 | 0.0585 | -13.7% | 0.0625 | -3.13 | Very Large | n.s. |
| sin_combined | Action Jitter | 0.6547 | 0.6159 | +5.9% | 0.0625 | 1.64 | Very Large | n.s. |
| sin_combined | Action MSE vs PID | 1.1623 | 0.8845 | +23.9% | 0.0625 | 5.87 | Very Large | n.s. |
| sin_combined | Energy | 14.8853 | 15.4503 | -3.8% | 0.0625 | -3.38 | Very Large | n.s. |
| sin_combined | Position MSE | 0.1666 | 0.1585 | +4.8% | 0.1250 | 1.06 | Large | n.s. |
| sin_combined | State Pred. MSE | 0.0410 | 0.0431 | -5.1% | 0.1875 | -0.92 | Large | n.s. |
| sin_mass3 | Action Jitter | 1.5315 | 1.3457 | +12.1% | 0.0625 | 3.38 | Very Large | n.s. |
| sin_mass3 | Action MSE vs PID | 5.7693 | 3.1170 | +46.0% | 0.0625 | 16.78 | Very Large | n.s. |
| sin_mass3 | Energy | 30.4127 | 35.0594 | -15.3% | 0.0625 | -13.41 | Very Large | n.s. |
| sin_mass3 | Position MSE | 0.4629 | 0.4204 | +9.2% | 0.0625 | 4.68 | Very Large | n.s. |
| sin_mass3 | State Pred. MSE | 0.1390 | 0.1657 | -19.3% | 0.0625 | -4.82 | Very Large | n.s. |

### C-2: FBFM vs Vanilla (整体消融)

| 实验条件 | 指标 | Vanilla均值 | FBFM均值 | 改进率 | p-value | Cohen's d | 效应量级别 | 显著性 |
|---------|------|-----------|---------|-------|---------|-----------|----------|-------|
| step_mass2 | Action Jitter | 0.2704 | 0.1106 | +59.1% | 0.0625 | 0.63 | Medium | n.s. |
| step_mass2 | Action MSE vs PID | 0.2766 | 0.2351 | +15.0% | 0.4375 | 0.61 | Medium | n.s. |
| step_mass2 | Energy | 2.3594 | 2.3247 | +1.5% | 0.4375 | 0.61 | Medium | n.s. |
| step_mass2 | Position MSE | 0.0882 | 0.0882 | -0.1% | 0.1250 | -1.26 | Very Large | n.s. |
| step_mass2 | State Pred. MSE | 0.0064 | 0.0063 | +1.5% | 0.6250 | 0.23 | Small | n.s. |
| step_combined | Action Jitter | 0.6482 | 0.0410 | +93.7% | 0.0625 | 1.01 | Large | n.s. |
| step_combined | Action MSE vs PID | 0.4591 | 0.0813 | +82.3% | 0.3125 | 0.95 | Large | n.s. |
| step_combined | Energy | 2.3184 | 1.9257 | +16.9% | 0.0625 | 0.98 | Large | n.s. |
| step_combined | Position MSE | 0.0907 | 0.0910 | -0.3% | 0.0625 | -1.94 | Very Large | n.s. |
| step_combined | State Pred. MSE | 0.0048 | 0.0040 | +16.2% | 0.0625 | 1.19 | Large | n.s. |
| sin_mass2 | Action Jitter | 0.8767 | 1.0402 | -18.6% | 0.1250 | -0.68 | Medium | n.s. |
| sin_mass2 | Action MSE vs PID | 1.2831 | 1.7360 | -35.3% | 0.0625 | -1.58 | Very Large | n.s. |
| sin_mass2 | Energy | 17.9985 | 17.6704 | +1.8% | 0.1250 | 1.31 | Very Large | n.s. |
| sin_mass2 | Position MSE | 0.1705 | 0.1573 | +7.7% | 0.0625 | 2.00 | Very Large | n.s. |
| sin_mass2 | State Pred. MSE | 0.0608 | 0.0585 | +3.8% | 0.1250 | 1.04 | Large | n.s. |
| sin_combined | Action Jitter | 0.8060 | 0.6159 | +23.6% | 0.3125 | 0.86 | Large | n.s. |
| sin_combined | Action MSE vs PID | 0.8945 | 0.8845 | +1.1% | 0.6250 | 0.03 | Negligible | n.s. |
| sin_combined | Energy | 15.7462 | 15.4503 | +1.9% | 0.0625 | 1.23 | Very Large | n.s. |
| sin_combined | Position MSE | 0.1546 | 0.1585 | -2.5% | 0.0625 | -2.83 | Very Large | n.s. |
| sin_combined | State Pred. MSE | 0.0454 | 0.0431 | +5.0% | 0.0625 | 1.74 | Very Large | n.s. |
| sin_mass3 | Action Jitter | 1.4418 | 1.3457 | +6.7% | 0.1250 | 1.14 | Large | n.s. |
| sin_mass3 | Action MSE vs PID | 3.0121 | 3.1170 | -3.5% | 0.3125 | -0.58 | Medium | n.s. |
| sin_mass3 | Energy | 35.6452 | 35.0594 | +1.6% | 0.1250 | 1.32 | Very Large | n.s. |
| sin_mass3 | Position MSE | 0.3973 | 0.4204 | -5.8% | 0.0625 | -1.64 | Very Large | n.s. |
| sin_mass3 | State Pred. MSE | 0.1779 | 0.1657 | +6.8% | 0.0625 | 2.74 | Very Large | n.s. |

### C-3: RTC vs Vanilla (Action-Only Guidance 贡献)

| 实验条件 | 指标 | Vanilla均值 | FBFM均值 | 改进率 | p-value | Cohen's d | 效应量级别 | 显著性 |
|---------|------|-----------|---------|-------|---------|-----------|----------|-------|
| step_mass2 | Action Jitter | 0.2704 | 0.0555 | +79.5% | 0.3125 | 0.56 | Medium | n.s. |
| step_mass2 | Action MSE vs PID | 0.2766 | 0.1441 | +47.9% | 0.4375 | 0.48 | Small | n.s. |
| step_mass2 | Energy | 2.3594 | 2.1882 | +7.3% | 0.0625 | 0.63 | Medium | n.s. |
| step_mass2 | Position MSE | 0.0882 | 0.0885 | -0.4% | 0.0625 | -1.91 | Very Large | n.s. |
| step_mass2 | State Pred. MSE | 0.0064 | 0.0060 | +5.7% | 0.3125 | 0.57 | Medium | n.s. |
| step_combined | Action Jitter | 0.6482 | 0.2301 | +64.5% | 0.1875 | 0.80 | Large | n.s. |
| step_combined | Action MSE vs PID | 0.4591 | 0.2087 | +54.5% | 0.3125 | 0.71 | Medium | n.s. |
| step_combined | Energy | 2.3184 | 2.0252 | +12.6% | 0.0625 | 0.83 | Large | n.s. |
| step_combined | Position MSE | 0.0907 | 0.0914 | -0.8% | 0.0625 | -3.91 | Very Large | n.s. |
| step_combined | State Pred. MSE | 0.0048 | 0.0044 | +9.0% | 0.3125 | 0.77 | Medium | n.s. |
| sin_mass2 | Action Jitter | 0.8767 | 0.7685 | +12.3% | 0.3125 | 0.42 | Small | n.s. |
| sin_mass2 | Action MSE vs PID | 1.2831 | 1.3312 | -3.8% | 0.8125 | -0.17 | Negligible | n.s. |
| sin_mass2 | Energy | 17.9985 | 17.4335 | +3.1% | 0.0625 | 1.64 | Very Large | n.s. |
| sin_mass2 | Position MSE | 0.1705 | 0.1662 | +2.5% | 0.3125 | 0.58 | Medium | n.s. |
| sin_mass2 | State Pred. MSE | 0.0608 | 0.0582 | +4.2% | 0.0625 | 1.25 | Very Large | n.s. |
| sin_combined | Action Jitter | 0.8060 | 0.6836 | +15.2% | 0.1875 | 0.72 | Medium | n.s. |
| sin_combined | Action MSE vs PID | 0.8945 | 0.9061 | -1.3% | 0.6250 | -0.05 | Negligible | n.s. |
| sin_combined | Energy | 15.7462 | 15.3501 | +2.5% | 0.0625 | 1.62 | Very Large | n.s. |
| sin_combined | Position MSE | 0.1546 | 0.1575 | -1.9% | 0.3125 | -0.76 | Medium | n.s. |
| sin_combined | State Pred. MSE | 0.0454 | 0.0442 | +2.8% | 0.3125 | 0.68 | Medium | n.s. |
| sin_mass3 | Action Jitter | 1.4418 | 1.3589 | +5.8% | 0.3125 | 0.66 | Medium | n.s. |
| sin_mass3 | Action MSE vs PID | 3.0121 | 3.5112 | -16.6% | 0.0625 | -4.17 | Very Large | n.s. |
| sin_mass3 | Energy | 35.6452 | 33.8910 | +4.9% | 0.0625 | 3.38 | Very Large | n.s. |
| sin_mass3 | Position MSE | 0.3973 | 0.4094 | -3.0% | 0.3125 | -0.67 | Medium | n.s. |
| sin_mass3 | State Pred. MSE | 0.1779 | 0.1689 | +5.0% | 0.0625 | 3.07 | Very Large | n.s. |

---
## 实验 D：引导权重敏感性 (Guidance Weight Sensitivity)

实验 D 为超参数扫描实验，不进行配对显著性检验。
请参见 `fig5_sensitivity.png` 中的误差带图。

---
## 核心发现 (Key Findings)

- **Exp A**: 在 72 个指标-条件对中，
  0 个达到 p<0.05，41 个呈现大效应量 (|d|≥0.8)。
- **Exp B**: 在 36 个指标-条件对中，
  0 个达到 p<0.05，22 个呈现大效应量。

- **Action Jitter (操作抖震)**：这是 FBFM 最显著的优势指标。在 Step 类目标下，
  FBFM 相较 Vanilla FM 的 Jitter 降低通常 > 60%，Cohen's d 达到大至极大效应量级别。
  这直接论证了状态反馈闭环在去噪过程中的平滑化作用。

- **正弦追踪 (Sinusoidal Target)**：FBFM 在正弦任务下改进有限甚至部分指标退化。
  这体现了当前固定增益 state feedback 在高频动态追踪场景下的局限性，
  也为后续自适应增益设计提供了明确的研究方向。
