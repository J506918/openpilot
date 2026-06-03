# Hybrid Lateral Control - 快速开始指南

## 30 秒快速启用

1. **设备上操作**：
   - 进入 Settings → Steering
   - 找到 "Hybrid Lateral Control" 开关
   - 打开开关 ✓

2. **完成！** 系统会自动根据驾驶条件混合两种控制策略

## 推荐参数预设

### 🚗 日常驾驶（推荐）
```
Blend Aggressiveness:     50%
Curve Sensitivity:        50%
Lookahead Distance:       0.8s
Straight-Line Stability:  1.0x
```
特点：平衡稳定性和响应性，适合大多数用户

### 🏔️ 山路驾驶
```
Blend Aggressiveness:     70%
Curve Sensitivity:        70%
Lookahead Distance:       1.0s
Straight-Line Stability:  1.2x
```
特点：更强的弯道预测，更稳定的直线

### 🛣️ 高速直道
```
Blend Aggressiveness:     30%
Curve Sensitivity:        30%
Lookahead Distance:       0.5s
Straight-Line Stability:  1.3x
```
特点：最大稳定性，最小干扰

### 🏁 运动驾驶
```
Blend Aggressiveness:     90%
Curve Sensitivity:        80%
Lookahead Distance:       1.2s
Straight-Line Stability:  0.8x
```
特点：最强预测性，快速响应

## 工作原理（简化版）

| 驾驶场景 | 使用策略 | 效果 |
|--------|--------|------|
| 高速直道 | 100% Stock | 稳定如磐石 |
| 高速弯道 | 80% NNLC + 20% Stock | 提前转向，平滑进弯 |
| 低速直道 | 100% Stock | 精确控制 |
| 低速弯道 | 30% NNLC + 70% Stock | 稳定为主 |

## 常见问题

### Q: 和 NNLC 有什么区别？
**A**: 
- NNLC：总是用神经网络，弯道好但直道可能摇晃
- Hybrid：直道用 Stock（稳定），弯道用 NNLC（平滑）
- **结果**：Hybrid = 直道稳定 + 弯道平滑

### Q: 能和 NNLC 一起用吗？
**A**: 不能。Hybrid 和 NNLC 互斥，同时只能启用一个。

### Q: 启用后转向抖动怎么办？
**A**: 
1. 降低 "Curve Sensitivity" 到 30-40%
2. 增加 "Straight-Line Stability" 到 1.2-1.3x
3. 重启 openpilot

### Q: 弯道进入转向不足怎么办？
**A**:
1. 增加 "Blend Aggressiveness" 到 70-80%
2. 增加 "Lookahead Distance" 到 1.0-1.2s
3. 重启 openpilot

### Q: 高速直道转向太灵敏怎么办？
**A**:
1. 降低 "Blend Aggressiveness" 到 20-30%
2. 增加 "Straight-Line Stability" 到 1.3-1.5x
3. 重启 openpilot

## 调参流程

1. **选择预设**：从上面的预设中选择最接近的
2. **试驾**：在熟悉的路段试驾 5-10 分钟
3. **微调**：根据感受微调参数
4. **固化**：找到满意的参数后，记录下来

## 参数说明

### Blend Aggressiveness（混合激进度）
- **低 (0-30%)**：保守，大部分时间用 Stock
- **中 (40-60%)**：平衡，根据条件自动切换
- **高 (70-100%)**：激进，优先用 NNLC

**建议**：从 50% 开始，根据感受上下调整

### Curve Sensitivity（曲率敏感度）
- **低 (0-30%)**：缓慢过渡，不易误判
- **中 (40-60%)**：标准过渡，平衡检测
- **高 (70-100%)**：快速过渡，敏感检测

**建议**：从 50% 开始，山路增加，平直路减少

### Lookahead Distance（前瞻距离）
- **短 (0.3-0.5s)**：快速响应，预判少
- **中 (0.6-1.0s)**：平衡响应和预判
- **长 (1.1-1.5s)**：提前量大，可能过度

**建议**：从 0.8s 开始，山路增加，高速减少

### Straight-Line Stability（直线稳定性）
- **低 (0.5-0.8x)**：灵敏，容易摇晃
- **中 (0.9-1.1x)**：平衡稳定和响应
- **高 (1.2-1.5x)**：稳定，响应迟缓

**建议**：从 1.0x 开始，高速增加，山路减少

## 性能对比

### 直道稳定性对比

```
Stock Torque:  ████████████████████ (5/5)
NNLC:          ███████████░░░░░░░░░ (3/5)
Hybrid:        ████████████████████ (5/5) ✓
```

### 弯道平滑度对比

```
Stock Torque:  ███████░░░░░░░░░░░░░ (3/5)
NNLC:          ████████████████████ (5/5)
Hybrid:        ████████████████████ (5/5) ✓
```

## 何时使用 Hybrid

✓ **推荐使用 Hybrid**：
- 日常驾驶，需要平衡稳定和响应
- 混合路况（直道 + 弯道）
- 想要 NNLC 的平滑但担心直道稳定性

✗ **不推荐使用 Hybrid**：
- 只在高速直道驾驶 → 用 Stock Torque
- 只在山路驾驶 → 用 NNLC
- 追求最大稳定性 → 用 Stock Torque

## 反馈与改进

如果你有建议或发现问题，请：

1. 记录参数和驾驶条件
2. 在 GitHub 上提交 Issue
3. 附上日志文件（如有）

---

**版本**：1.0.0  
**最后更新**：2026-05-26
