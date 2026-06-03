"""
Hybrid Lateral Control - Combines NNLC Foresight with Stock Stability
Copyright (c) 2026, sunnypilot contributors

This controller blends:
- NNLC's predictive lookahead capability for smooth curve navigation
- Stock torque controller's rock-solid stability on straights
- Adaptive blending based on speed, curvature, and lateral acceleration

Key features:
1. Predictive feedforward using model_v2 (NNLC-style)
2. Stable PID error correction (stock-style)
3. Adaptive gain scheduling based on driving conditions
4. Friction compensation for consistent steering feel
"""

import math
import numpy as np
from collections import deque

from cereal import log
from opendbc.car.lateral import FRICTION_THRESHOLD, get_friction
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.pid import PIDController
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.selfdrive.modeld.constants import ModelConstants

# ─── PID Tuning ────────────────────────────────────────────────────────────
# More aggressive than stock to handle curve entry
KP = 1.2
KI = 0.35

# Speed-dependent gains for smooth transitions
INTERP_SPEEDS = [1, 1.5, 2.0, 3.0, 5, 7.5, 10, 15, 30]
KP_INTERP = [280, 140, 75, 35, 13, 6.5, 4.0, 2.2, KP]

# ─── Predictive Lookahead ──────────────────────────────────────────────────
LP_FILTER_CUTOFF_HZ = 1.2
JERK_LOOKAHEAD_SECONDS = 0.19
JERK_GAIN = 0.35  # Slightly higher than stock for better curve anticipation

LAT_ACCEL_REQUEST_BUFFER_SECONDS = 1.0
LOOKAHEAD_TIME = 0.8  # How far ahead to look for curve prediction

# ─── Blending Parameters ───────────────────────────────────────────────────
# Blend NNLC-style feedforward with stock-style error correction
# At low speed: use stock (stable)
# At high speed: use NNLC (predictive)
BLEND_SPEED_LOW = 5.0   # m/s - below this, use mostly stock
BLEND_SPEED_HIGH = 25.0  # m/s - above this, use mostly NNLC

# Curvature-based blending: higher curvature = more NNLC
BLEND_CURVATURE_LOW = 0.001   # rad/m
BLEND_CURVATURE_HIGH = 0.01   # rad/m

# Lateral acceleration blending: higher lat accel = more NNLC
BLEND_LAT_ACCEL_LOW = 0.5    # m/s^2
BLEND_LAT_ACCEL_HIGH = 3.0   # m/s^2

VERSION = 2


class HybridLateralControl(LatControl):
  """
  Hybrid lateral controller combining NNLC foresight with stock stability.
  
  The controller uses:
  1. Predictive model lookahead to anticipate curves (NNLC-style)
  2. Stable PID error correction (stock-style)
  3. Adaptive blending between the two strategies
  """
  
  def __init__(self, CP, CP_SP, CI, dt):
    super().__init__(CP, CP_SP, CI, dt)
    
    self.torque_params = CP.lateralTuning.torque.as_builder()
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.lateral_accel_from_torque = CI.lateral_accel_from_torque()
    
    # PID controller with speed-dependent gains
    self.pid = PIDController([INTERP_SPEEDS, KP_INTERP], KI, rate=1/self.dt)
    self.update_limits()
    
    # Steering angle deadzone
    self.steering_angle_deadzone_deg = self.torque_params.steeringAngleDeadzoneDeg
    
    # Lateral acceleration buffer for delay compensation
    self.lat_accel_request_buffer_len = int(LAT_ACCEL_REQUEST_BUFFER_SECONDS / self.dt)
    self.lat_accel_request_buffer = deque([0.] * self.lat_accel_request_buffer_len, 
                                          maxlen=self.lat_accel_request_buffer_len)
    
    # Lookahead frames for jerk prediction
    self.lookahead_frames = int(JERK_LOOKAHEAD_SECONDS / self.dt)
    self.jerk_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * LP_FILTER_CUTOFF_HZ), self.dt)
    
    # Model state for predictive lookahead
    self.model_v2 = None
    self.model_valid = False
    
    # Blending state
    self.blend_factor = 0.0  # 0 = stock, 1 = NNLC
    self.prev_blend_factor = 0.0
    
    # Smoothing filter for blend factor transitions
    self.blend_filter = FirstOrderFilter(0.0, 0.5, dt)
    
    # Predictive feedforward accumulator
    self.predictive_ff = 0.0
    
    # Logging
    self.last_blend_factor = 0.0
    self.last_predictive_ff = 0.0

  def update_limits(self):
    self.pid.set_limits(self.lateral_accel_from_torque(self.steer_max, self.torque_params),
                        self.lateral_accel_from_torque(-self.steer_max, self.torque_params))

  def update_model_v2(self, model_v2):
    """Update model predictions for lookahead."""
    self.model_v2 = model_v2
    self.model_valid = self.model_v2 is not None and len(self.model_v2.orientation.x) >= 16

  def _compute_blend_factor(self, CS, desired_curvature, desired_lateral_accel):
    """
    Compute adaptive blending factor between stock (0) and NNLC (1).
    
    Higher blend factor = more predictive NNLC behavior
    Lower blend factor = more stable stock behavior
    """
    # Speed-based blending
    speed_blend = np.interp(CS.vEgo, [BLEND_SPEED_LOW, BLEND_SPEED_HIGH], [0.0, 1.0])
    
    # Curvature-based blending (more curve = more NNLC)
    curvature_blend = np.interp(abs(desired_curvature), 
                                [BLEND_CURVATURE_LOW, BLEND_CURVATURE_HIGH], 
                                [0.0, 1.0])
    
    # Lateral acceleration blending (more aggressive = more NNLC)
    lat_accel_blend = np.interp(abs(desired_lateral_accel),
                                [BLEND_LAT_ACCEL_LOW, BLEND_LAT_ACCEL_HIGH],
                                [0.0, 1.0])
    
    # Combine blending factors: use the maximum to be more aggressive in curves
    blend = max(speed_blend * 0.4, curvature_blend * 0.8, lat_accel_blend * 0.6)
    
    # Smooth transitions with filter
    blend = self.blend_filter.update(blend)
    
    return np.clip(blend, 0.0, 1.0)

  def _compute_predictive_feedforward(self, CS, desired_lateral_accel):
    """
    Compute predictive feedforward using model lookahead position (NNLC-style).

    Uses model_v2.position.y → curvature → lateral acceleration.
    position.y is geometric — independent of controller assumptions,
    unlike acceleration.y which contains V0 PID training bias.
    """
    if not self.model_valid or self.model_v2 is None:
      return 0.0

    try:
      n = len(self.model_v2.position.y)
      # T_IDXS from ModelConstants: quadratic time indices for each model frame
      t_idxs = ModelConstants.T_IDXS
      # Find index closest to LOOKAHEAD_TIME
      lookahead_idx = int(np.clip(
        np.searchsorted(t_idxs, LOOKAHEAD_TIME, side='left'), 2, n - 2))

      # Use 3-point central difference for curvature: κ ≈ d²y/dx²
      # dx between frames: vEgo * (t[i+1] - t[i-1]) / 2
      dt = (t_idxs[lookahead_idx + 1] - t_idxs[lookahead_idx - 1]) / 2
      dx = CS.vEgo * dt
      if dx < 1e-6:
        return 0.0

      y_prev = self.model_v2.position.y[lookahead_idx - 1]
      y_curr = self.model_v2.position.y[lookahead_idx]
      y_next = self.model_v2.position.y[lookahead_idx + 1]

      curvature = (y_prev - 2 * y_curr + y_next) / (dx * dx)

      # Convert curvature to physical lateral acceleration: a_y = κ · v²
      predicted_lat_accel = curvature * CS.vEgo ** 2

      lat_accel_delta = predicted_lat_accel - desired_lateral_accel
      predictive_ff = lat_accel_delta * 0.3  # Tuning factor

      return predictive_ff
    except (IndexError, AttributeError, TypeError):
      return 0.0

  def _compute_stock_feedforward(self, CS, desired_lateral_accel, roll_compensation, 
                                 lateral_accel_deadzone, error, desired_lateral_jerk):
    """
    Compute stock-style feedforward (simple and stable).
    """
    # Gravity-adjusted future lateral accel
    ff = desired_lateral_accel - roll_compensation
    
    # Friction compensation with jerk feedforward term
    ff += get_friction(error + JERK_GAIN * desired_lateral_jerk, lateral_accel_deadzone, 
                      FRICTION_THRESHOLD, self.torque_params)
    
    return ff

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, 
             calibrated_pose, curvature_limited, lat_delay):
    """
    Main control update loop.
    
    Blends between stock stable control and NNLC predictive control.
    """
    pid_log = log.ControlsState.LateralTorqueState.new_message()
    pid_log.version = VERSION
    
    if not active:
      return 0.0, 0.0, pid_log
    
    # ─── Measurements ──────────────────────────────────────────────────────
    measured_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), 
                                           CS.vEgo, params.roll)
    measurement = measured_curvature * CS.vEgo ** 2
    
    desired_lateral_accel = desired_curvature * CS.vEgo ** 2
    self.lat_accel_request_buffer.append(desired_lateral_accel)
    
    roll_compensation = params.roll * ACCELERATION_DUE_TO_GRAVITY
    curvature_deadzone = abs(VM.calc_curvature(math.radians(self.steering_angle_deadzone_deg), 
                                               CS.vEgo, 0.0))
    lateral_accel_deadzone = curvature_deadzone * CS.vEgo ** 2
    
    # ─── Delay Compensation ────────────────────────────────────────────────
    delay_frames = int(np.clip(lat_delay / self.dt + 1, 1, self.lat_accel_request_buffer_len))
    expected_lateral_accel = self.lat_accel_request_buffer[-delay_frames]
    
    # ─── Error Computation ─────────────────────────────────────────────────
    setpoint = expected_lateral_accel
    error = setpoint - measurement
    pid_log.error = float(error)
    
    # ─── Jerk Prediction ───────────────────────────────────────────────────
    lookahead_idx = int(np.clip(-delay_frames + self.lookahead_frames, 
                                -self.lat_accel_request_buffer_len + 1, -2))
    raw_lateral_jerk = (self.lat_accel_request_buffer[lookahead_idx + 1] - 
                       self.lat_accel_request_buffer[lookahead_idx - 1]) / (2 * self.dt)
    desired_lateral_jerk = self.jerk_filter.update(raw_lateral_jerk)
    
    # ─── Adaptive Blending ─────────────────────────────────────────────────
    self.blend_factor = self._compute_blend_factor(CS, desired_curvature, desired_lateral_accel)
    
    # ─── Feedforward Computation ───────────────────────────────────────────
    stock_ff = self._compute_stock_feedforward(CS, desired_lateral_accel, roll_compensation,
                                              lateral_accel_deadzone, error, desired_lateral_jerk)

    predictive_ff = self._compute_predictive_feedforward(CS, desired_lateral_accel)

    # Blend the two feedforward strategies
    ff = stock_ff * (1.0 - self.blend_factor) + predictive_ff * self.blend_factor

    # ─── PID Control ───────────────────────────────────────────────────────
    freeze_integrator = steer_limited_by_safety or CS.steeringPressed or CS.vEgo < 2.0
    output_lataccel = self.pid.update(pid_log.error, speed=CS.vEgo, 
                                      feedforward=ff, freeze_integrator=freeze_integrator)
    output_torque = self.torque_from_lateral_accel(output_lataccel, self.torque_params)
    
    # ─── Logging ────────────────────────────────────────────────────────────
    pid_log.active = True
    pid_log.p = float(self.pid.p)
    pid_log.i = float(self.pid.i)
    pid_log.d = float(self.pid.d)
    pid_log.f = float(self.pid.f)
    pid_log.output = float(-output_torque)
    pid_log.actualLateralAccel = float(measurement)
    pid_log.desiredLateralAccel = float(setpoint)
    pid_log.desiredLateralJerk = float(desired_lateral_jerk)
    pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, 
                                                    CS, steer_limited_by_safety, curvature_limited))
    
    # Store for debugging
    self.last_blend_factor = self.blend_factor
    self.last_predictive_ff = predictive_ff
    
    return -output_torque, 0.0, pid_log
