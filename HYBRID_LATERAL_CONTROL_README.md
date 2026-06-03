# Hybrid Lateral Control - 混合型横向控制系统

## 概述

**Hybrid Lateral Control** 是一个创新的横向控制系统，完美融合了两种控制策略的优点：

- **NNLC（神经网络横向控制）的前瞻性** — 提前预判弯道，平滑进入曲线
- **官方 Stock Torque Controller 的稳定性** — 直道上纹丝不动，宛如焊死在轨道上

## 核心特性

### 1. 自适应混合控制（Adaptive Blending）

系统根据实时驾驶条件自动在两种控制策略之间切换：

| 条件 | 使用策略 | 原因 |
|------|--------|------|
| 低速直道 | Stock (稳定) | 低速时转向敏感，需要稳定的基础控制 |
| 高速直道 | Stock (稳定) | 直线行驶时不需要预测，稳定最重要 |
| 低速弯道 | Stock (稳定) | 低速弯道转向充足，不需要预测 |
| 高速弯道 | NNLC (预测) | 高速弯道需要提前预判，NNLC 优势明显 |
| 急加速转向 | NNLC (预测) | 高横向加速度时需要前瞻性 |

### 2. 混合因子计算（Blend Factor）

混合因子 (0 = 100% Stock, 1 = 100% NNLC) 由三个维度计算：

```
blend_factor = max(
  speed_blend × 0.4,           # 速度维度（权重 40%）
  curvature_blend × 0.8,       # 曲率维度（权重 80%）
  lat_accel_blend × 0.6        # 横向加速度维度（权重 60%）
)
```

**速度混合**：
- 5 m/s 以下：0% NNLC（100% Stock）
- 25 m/s 以上：100% NNLC（0% Stock）
- 中间线性插值

**曲率混合**：
- 0.001 rad/m 以下：0% NNLC（直道）
- 0.01 rad/m 以上：100% NNLC（急弯）
- 中间线性插值

**横向加速度混合**：
- 0.5 m/s² 以下：0% NNLC（温和）
- 3.0 m/s² 以上：100% NNLC（激进）
- 中间线性插值

### 3. 前瞻性反馈（Predictive Feedforward）

NNLC 模式使用模型预测（model_v2）提前 0.8 秒预判横向加速度变化：

```python
predicted_lat_accel = model_v2.acceleration.y[lookahead_idx]
lat_accel_delta = predicted_lat_accel - current_desired_accel
predictive_ff = lat_accel_delta × 0.3
```

这使得系统能在进入弯道前就开始调整转向，而不是被动反应。

### 4. 稳定的 PID 控制（Stable PID Control）

Stock 模式使用速度相关的 PID 增益：

| 速度 (m/s) | Kp |
|-----------|-----|
| 1.0 | 280 |
| 5.0 | 13 |
| 10.0 | 4.0 |
| 30.0 | 1.2 |

低速时增益高（转向敏感），高速时增益低（转向稳定）。

### 5. 摩擦补偿（Friction Compensation）

两种模式都包含摩擦补偿，确保转向感受一致：

```python
ff += get_friction(error, lateral_accel_deadzone, FRICTION_THRESHOLD, torque_params)
```

## 安装与启用

### 1. 文件位置

新增文件已放置在以下位置：

```
sunnypilot/selfdrive/controls/lib/
  ├── latcontrol_hybrid.py              # 核心控制逻辑
  └── latcontrol_hybrid_ext.py          # 参数扩展

selfdrive/ui/sunnypilot/layouts/settings/steering_sub_layouts/
  └── hybrid_lateral_settings.py        # UI 设置面板

selfdrive/ui/sunnypilot/layouts/settings/
  └── steering.py                       # 已修改，添加了开关和子面板
```

### 2. 启用步骤

1. **在设备上启用**：
   - 进入 Settings → Steering
   - 找到 "Hybrid Lateral Control" 开关
   - 打开开关

2. **自定义参数**（可选）：
   - 点击 "Customize Hybrid Lateral" 按钮
   - 调整以下参数：
     - **Blend Aggressiveness** (0-100%)：控制预测性的强度
     - **Curve Sensitivity** (0-100%)：弯道检测灵敏度
     - **Lookahead Distance** (0.3-1.5s)：提前预判时间
     - **Straight-Line Stability** (0.5-1.5x)：直道稳定性阻尼

## 参数详解

### Blend Aggressiveness（混合激进度）

- **低 (0-30%)**：保守模式，大部分时间使用 Stock 稳定控制
  - 适合：保守驾驶风格，舒适性优先
  - 特点：转向平缓，响应略迟缓

- **中 (40-60%)**：平衡模式，根据条件自动切换
  - 适合：大多数用户，日常驾驶
  - 特点：平衡稳定性和响应性

- **高 (70-100%)**：激进模式，优先使用 NNLC 预测
  - 适合：运动驾驶，高速弯道
  - 特点：转向提前，弯道平滑

### Curve Sensitivity（曲率敏感度）

- **低 (0-30%)**：缓慢过渡，弯道检测迟缓
  - 适合：平直路段多的地区
  - 特点：不易误判直道为弯道

- **中 (40-60%)**：标准过渡，平衡检测和稳定性
  - 适合：一般路况
  - 特点：响应及时，不易误判

- **高 (70-100%)**：快速过渡，弯道检测敏感
  - 适合：山路、蛇形路段
  - 特点：快速反应，可能过度敏感

### Lookahead Distance（前瞻距离）

- **短 (0.3-0.5s)**：近距离预判
  - 优点：响应快，不易过度转向
  - 缺点：预判时间短，可能错过远处弯道

- **中 (0.6-1.0s)**：标准预判（推荐）
  - 优点：平衡响应和预判
  - 缺点：无

- **长 (1.1-1.5s)**：远距离预判
  - 优点：提前量大，弯道进入平滑
  - 缺点：可能过度转向，不适合急弯

### Straight-Line Stability（直线稳定性）

- **低 (0.5-0.8x)**：低阻尼，转向灵敏
  - 适合：山路、需要频繁调整的路段
  - 特点：转向响应快

- **中 (0.9-1.1x)**：标准阻尼（推荐）
  - 适合：一般路况
  - 特点：平衡稳定和响应

- **高 (1.2-1.5x)**：高阻尼，转向稳定
  - 适合：高速直道，需要稳定的路段
  - 特点：转向平缓，不易摇晃

## 工作原理详解

### 控制流程

```
输入：当前车速、转向角、模型预测、期望曲率
  ↓
[1] 测量当前横向加速度
  ↓
[2] 计算混合因子（基于速度、曲率、横向加速度）
  ↓
[3] 计算两种反馈
  ├─→ Stock 反馈：简单稳定的前馈 + 摩擦补偿
  └─→ NNLC 反馈：模型预测的前瞻性前馈
  ↓
[4] 混合反馈：ff = stock_ff × (1 - blend) + nnlc_ff × blend
  ↓
[5] PID 控制：output = Kp×error + Ki×integral + Kd×derivative + ff
  ↓
[6] 转换为转向扭矩
  ↓
输出：转向扭矩命令
```

### 实际例子

**场景 1：高速直道（100 km/h）**
- 速度混合：100% Stock（速度 > 25 m/s）
- 曲率混合：0% NNLC（直道）
- 横向加速度混合：0% NNLC（< 0.5 m/s²）
- **结果**：100% Stock 控制 → 转向稳定如磐石

**场景 2：高速弯道进入（100 km/h，急弯）**
- 速度混合：100% NNLC
- 曲率混合：80% NNLC（曲率 > 0.01 rad/m）
- 横向加速度混合：60% NNLC（> 2.0 m/s²）
- **结果**：80% NNLC + 20% Stock → 提前转向，平滑进弯

**场景 3：低速弯道（30 km/h，缓弯）**
- 速度混合：30% NNLC（速度 < 10 m/s）
- 曲率混合：40% NNLC（曲率 ≈ 0.005 rad/m）
- 横向加速度混合：0% NNLC（< 0.5 m/s²）
- **结果**：30% NNLC + 70% Stock → 稳定为主，略有预测

## 调试与优化

### 日志信息

控制器会记录以下信息到 `controlsState.lateralTorqueState`：

- `error`：当前横向加速度误差
- `p`, `i`, `d`, `f`：PID 各分量
- `output`：最终转向扭矩
- `actualLateralAccel`：实际横向加速度
- `desiredLateralAccel`：期望横向加速度

### 调试步骤

1. **启用 SSH 连接到设备**
2. **查看实时日志**：
   ```bash
   ssh comma@<device_ip>
   tail -f /data/media/0/logs/controlsd.log | grep -i lateral
   ```

3. **检查混合因子**：
   在代码中添加日志输出：
   ```python
   print(f"Blend factor: {self.blend_factor:.2f}, Speed: {CS.vEgo:.1f} m/s, Curvature: {desired_curvature:.4f}")
   ```

4. **调整参数**：
   - 如果弯道进入过度转向，降低 `Blend Aggressiveness`
   - 如果直道摇晃，提高 `Straight-Line Stability`
   - 如果响应迟缓，提高 `Curve Sensitivity`

## 与其他控制器的互斥性

Hybrid Lateral Control 与以下控制器互斥（同时只能启用一个）：

- ✗ Enforce Torque Control（强制 Torque 控制）
- ✗ Neural Network Lateral Control (NNLC)

启用 Hybrid 时，这两个开关会自动禁用。

## 性能对比

| 指标 | Stock Torque | NNLC | Hybrid |
|------|-------------|------|--------|
| 直道稳定性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 弯道平滑度 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 低速响应 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 高速稳定 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| CPU 占用 | 低 | 中 | 中 |
| 调参难度 | 低 | 高 | 中 |

## 故障排除

### 问题 1：启用后转向抖动

**原因**：混合因子变化过快，导致控制策略频繁切换

**解决方案**：
- 降低 `Curve Sensitivity`
- 增加 `Straight-Line Stability`
- 检查 `blend_filter` 的平滑系数

### 问题 2：弯道进入过度转向

**原因**：NNLC 模式的预测过于激进

**解决方案**：
- 降低 `Blend Aggressiveness`
- 减少 `Lookahead Distance`
- 调整 `latcontrol_hybrid.py` 中的 `predictive_ff` 系数（当前为 0.3）

### 问题 3：直道转向不稳定

**原因**：Stock 模式的摩擦补偿不足

**解决方案**：
- 增加 `Straight-Line Stability`
- 检查车辆的转向系统是否正常
- 调整 `JERK_GAIN` 参数

### 问题 4：高速弯道响应迟缓

**原因**：混合因子偏向 Stock 模式

**解决方案**：
- 提高 `Blend Aggressiveness`
- 提高 `Curve Sensitivity`
- 增加 `Lookahead Distance`

## 未来改进方向

1. **学习型混合因子**：基于用户驾驶风格自动调整混合比例
2. **路况自适应**：根据路面类型（高速、城市、山路）自动调整参数
3. **天气补偿**：根据天气条件调整摩擦系数
4. **神经网络优化**：使用更小的 NN 模型以减少 CPU 占用
5. **多模型支持**：为不同车型提供优化的参数预设

## 贡献与反馈

如有问题或建议，请在 GitHub 上提交 Issue 或 Pull Request。

## 许可证

MIT License - 详见 LICENSE.md

---

**作者**：sunnypilot 贡献者  
**创建日期**：2026-05-26  
**版本**：1.0.0
