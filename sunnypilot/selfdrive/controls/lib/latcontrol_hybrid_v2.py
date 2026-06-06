"""
Hybrid Lateral Control V2 — Model-Driven Torque with Slow Bias Correction
Copyright (c) 2026, sunnypilot contributors

Architecture:
  主通道（前馈，占 90%+）：
    model_v2.position.y[前瞻点] → 曲率 κ → a_y = κ·v²
    → torque_from_lateral_accel(a_y, params) + friction
    → 直接输出，零 PID

  辅助通道（偏置修正，占 < 10%）：
    实测曲率 vs 延迟对齐的预测曲率 → 极慢积分（~30 秒时间常数）
    → 只在有系统偏置（路面倾斜、胎压不均）时慢慢拉回来

Key differences from V1 (PID-blending Hybrid):
  - 零 PID：不追瞬态误差，不产生 overshoot/震荡
  - 主通道全前馈：model 画了线，照着走
  - 偏置修正秒级响应：不会追瞬态，只抗漂移
"""

import math
import numpy as np
from collections import deque

from cereal import log
from opendbc.car.lateral import get_friction
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.selfdrive.modeld.constants import ModelConstants

# ─── Tuning Constants ─────────────────────────────────────────────────────
LOOKAHEAD_TIME = 0.8          # seconds — default model lookahead
FRICTION_THRESHOLD = 0.3      # lateral accel deadzone for friction

# Bias correction — seconds-level time constant
# At 30 Hz, gain=0.001 → ~33 second time constant
BIAS_GAIN = 0.001
BIAS_MAX = 0.003              # max curvature bias (~0.17° at 15 m/s → ~0.07 m/s² lat accel)

# Low-pass filter for model curvature (smooth out frame-to-frame jitter)
CURVATURE_LP_FC = 2.0         # Hz — higher = more responsive, lower = smoother

VERSION = 3


class HybridLateralControlV2(LatControl):
  """Model-driven lateral control: direct torque mapping from model predictions."""

  def __init__(self, CP, CP_SP, CI, dt, ext=None):
    super().__init__(CP, CP_SP, CI, dt)

    # Act as own extension for controlsd's extension.update_model_v2() call
    self.extension = self

    self.hybrid_ext = ext

    self.torque_params = CP.lateralTuning.torque.as_builder()
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.lateral_accel_from_torque = CI.lateral_accel_from_torque()

    # Model state
    self.model_v2 = None
    self.model_valid = False

    # Low-pass filter for curvature
    self.curvature_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * CURVATURE_LP_FC), dt)

    # Buffer for delay-aligned bias comparison
    # Store predicted curvatures so we can compare against delayed measurement
    self.buffer_len = max(int(1.5 / dt), 30)  # ~1.5 seconds buffer
    self.curvature_buffer = deque(maxlen=self.buffer_len)

    # Slow bias correction
    self.bias_curvature = 0.0  # accumulated bias (rad/m)
    self.bias_filter = FirstOrderFilter(0.0, 30.0, dt)  # 30s time constant smoothing

  def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction):
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction

  def update_limits(self):
    pass  # No PID — no limits needed

  def update_lateral_lag(self, lag):
    pass

  def update_model_v2(self, model_v2):
    self.model_v2 = model_v2
    self.model_valid = self.model_v2 is not None and len(self.model_v2.position.y) >= 16

  def _compute_model_curvature(self, v_ego, lookahead_time):
    """Compute curvature from model position.y at the lookahead point.

    Uses 3-point central difference on model position.y for curvature: κ ≈ d²y/dx²
    """
    if not self.model_valid or self.model_v2 is None:
      return 0.0

    try:
      n = len(self.model_v2.position.y)
      t_idxs = ModelConstants.T_IDXS

      # Find the model frame closest to the desired lookahead time
      idx = int(np.clip(np.searchsorted(t_idxs, lookahead_time, side='left'), 1, n - 2))

      # 3-point central difference for curvature
      dt = (t_idxs[idx + 1] - t_idxs[idx - 1]) / 2
      dx = v_ego * dt
      if dx < 1e-6:
        return 0.0

      y_prev = self.model_v2.position.y[idx - 1]
      y_curr = self.model_v2.position.y[idx]
      y_next = self.model_v2.position.y[idx + 1]

      curvature = (y_prev - 2 * y_curr + y_next) / (dx * dx)
      return curvature
    except (IndexError, AttributeError, TypeError):
      return 0.0

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature,
             calibrated_pose, curvature_limited, lat_delay):
    """Main control update — model-driven feedforward with slow bias correction."""
    pid_log = log.ControlsState.LateralTorqueState.new_message()
    pid_log.version = VERSION

    if not active:
      self.bias_curvature = 0.0
      self.curvature_filter.x = 0.0
      return 0.0, 0.0, pid_log

    # ─── Lookahead time ─────────────────────────────────────────────────
    lookahead_time = LOOKAHEAD_TIME
    if self.hybrid_ext:
      lookahead_time = self.hybrid_ext.get_lookahead_time()

    # ─── Main channel: model curvature → direct torque ─────────────────
    raw_model_curvature = self._compute_model_curvature(CS.vEgo, lookahead_time)
    model_curvature = self.curvature_filter.update(raw_model_curvature)

    predicted_lat_accel = model_curvature * CS.vEgo ** 2
    base_torque = self.torque_from_lateral_accel(predicted_lat_accel, self.torque_params)

    # ─── Roll compensation ──────────────────────────────────────────────
    # latAccelOffset corrects for device mount misalignment relative to car roll
    roll_compensation = params.roll * ACCELERATION_DUE_TO_GRAVITY
    roll_torque = self.torque_from_lateral_accel(
      -roll_compensation - self.torque_params.latAccelOffset, self.torque_params)

    # ─── Friction compensation ──────────────────────────────────────────
    measured_curvature = -VM.calc_curvature(
      math.radians(CS.steeringAngleDeg - params.angleOffsetDeg),
      CS.vEgo, params.roll)
    measured_lat_accel = measured_curvature * CS.vEgo ** 2

    # Friction based on error between predicted and actual
    steering_angle_deadzone_deg = self.torque_params.steeringAngleDeadzoneDeg
    curvature_deadzone = abs(VM.calc_curvature(
      math.radians(steering_angle_deadzone_deg), CS.vEgo, 0.0))
    lateral_accel_deadzone = curvature_deadzone * CS.vEgo ** 2

    lat_accel_error = predicted_lat_accel - measured_lat_accel
    friction_torque = get_friction(
      lat_accel_error, lateral_accel_deadzone,
      FRICTION_THRESHOLD, self.torque_params)

    # ─── Slow bias correction ──────────────────────────────────────────
    # Buffer predicted curvature each frame
    self.curvature_buffer.append(model_curvature)

    # Compare current measurement against time-aligned past prediction
    delay_frames = int(np.clip(lat_delay / self.dt, 1, self.buffer_len))
    if len(self.curvature_buffer) > delay_frames and self.model_valid:
      old_predicted = self.curvature_buffer[-delay_frames]
      # Raw bias: how much actual curvature differs from what was predicted
      raw_bias = measured_curvature - old_predicted
      # Extremely slow accumulation
      self.bias_curvature += raw_bias * BIAS_GAIN
      self.bias_curvature = np.clip(self.bias_curvature, -BIAS_MAX, BIAS_MAX)

    # Apply filtered bias as lateral accel correction → torque
    bias_lat_accel = self.bias_filter.update(self.bias_curvature) * CS.vEgo ** 2
    bias_torque = self.torque_from_lateral_accel(bias_lat_accel, self.torque_params)

    # ─── Final output ───────────────────────────────────────────────────
    output_torque = base_torque + friction_torque + bias_torque + roll_torque

    # Safety: don't exceed steer_max, and don't apply torque when safety-limited
    if steer_limited_by_safety:
      output_torque = 0.0
    else:
      output_torque = np.clip(output_torque, -self.steer_max, self.steer_max)

    # ─── Logging ────────────────────────────────────────────────────────
    pid_log.active = True
    pid_log.p = float(base_torque)
    pid_log.i = float(bias_torque)
    pid_log.f = float(friction_torque)
    pid_log.output = float(-output_torque)
    pid_log.actualLateralAccel = float(measured_lat_accel)
    pid_log.desiredLateralAccel = float(predicted_lat_accel)
    pid_log.saturated = bool(abs(output_torque) >= self.steer_max * 0.99)

    return -output_torque, 0.0, pid_log
