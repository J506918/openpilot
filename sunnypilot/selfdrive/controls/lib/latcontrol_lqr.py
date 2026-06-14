"""
LQR Lateral Control — Physics Feedforward + Bounded State Feedback
Copyright (c) 2026, sunnypilot contributors

Architecture:
  lateralManeuverPlan.desiredCurvature → physics feedforward → torque (80%)
  + modelV2 position.y / orientation.z → bounded state feedback (15%)
  + curvature rate → lead compensation (5%)
  + roll + friction compensation

No PID integral — LQR-style state feedback handles steady-state error
without integral windup. Feedforward dominates, feedback is bounded.

Design reference:
  state feedback gains derived from bicycle model lateral dynamics,
  clipped to ±2.0 m/s² to prevent overshoot on large errors.
  curvature rate lead compensates for actuator delay (~0.25s lead).
"""

import math
import numpy as np

from cereal import log
from opendbc.car.lateral import get_friction
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.selfdrive.modeld.constants import ModelConstants

# ─── State Feedback Gains (speed-dependent) ─────────────────────────────
# K_lat: lateral error (m) → lat accel correction (m/s²)
# Higher at low speed (more authority), lower at high speed (stability)
LQR_K_LAT_BP = [5.0, 10.0, 15.0, 25.0, 35.0]    # m/s
LQR_K_LAT_V  = [2.5,  1.8,  1.2,  0.7,  0.5]    # (m/s²)/m

# K_heading: heading error (rad) → lat accel correction (m/s²)
# Scaled by vEgo: heading_error * vEgo = lateral velocity error
LQR_K_PSI_BP = [5.0, 10.0, 15.0, 25.0, 35.0]    # m/s
LQR_K_PSI_V  = [2.0,  1.5,  1.0,  0.6,  0.4]    # (m/s²)/(rad·m/s)

# ─── Tuning ─────────────────────────────────────────────────────────────
LQR_LOOKAHEAD_S = 0.4          # seconds ahead for state feedback
LQR_FB_LAT_LIMIT = 2.0         # ±m/s², max feedback correction
LQR_CURV_RATE_GAIN = 0.25      # curvature rate feedforward gain (s)
LQR_CURV_RATE_LP = 3.0         # Hz, low-pass for curvature rate
FRICTION_THRESHOLD = 0.3

# Deadband crossing: minimum torque factor relative to friction.
# Friction * factor must exceed EPS torque deadband (STEER_THRESHOLD/STEER_MAX).
# Honda Accord 11G: friction=0.17, deadband=0.234 → factor >= 1.38
DEADBAND_FACTOR = 1.5

VERSION = 90


def _compute_lookahead_idx(lookahead_s: float) -> int:
  """Find the modelV2 index closest to lookahead_s using T_IDXS spacing."""
  idx = int(np.searchsorted(ModelConstants.T_IDXS[:CONTROL_N], lookahead_s))
  return min(max(idx, 1), CONTROL_N - 1)  # at least 1, at most CONTROL_N-1


class LatControlLQR(LatControl):
  """LQR-style lateral controller: feedforward-dominant, no integral."""

  def __init__(self, CP, CP_SP, CI, dt):
    super().__init__(CP, CP_SP, CI, dt)

    # Act as own extension for controlsd's extension.update_model_v2() call
    self.extension = self

    self.torque_params = CP.lateralTuning.torque.as_builder()
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.lateral_accel_from_torque = CI.lateral_accel_from_torque()

    # Model state (set by update_model_v2)
    self.model_v2 = None
    self.model_valid = False

    # Pre-compute lookahead index from T_IDXS (non-uniform spacing)
    self._lookahead_idx = _compute_lookahead_idx(LQR_LOOKAHEAD_S)

    # Curvature rate low-pass filter
    self._curv_rate_filter = FirstOrderFilter(
      0.0, 1 / (2 * np.pi * LQR_CURV_RATE_LP), dt
    )
    self._prev_desired_curvature = 0.0
    self._first_active_frame = True
    self._prev_output_torque = 0.0

  def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction):
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction

  def update_limits(self):
    pass

  def update_lateral_lag(self, lag):
    pass

  def reset(self):
    super().reset()
    self._curv_rate_filter.x = 0.0
    self._prev_desired_curvature = 0.0
    self._first_active_frame = True
    self._prev_output_torque = 0.0

  def update_model_v2(self, model_v2):
    self.model_v2 = model_v2
    self.model_valid = (
      self.model_v2 is not None
      and len(self.model_v2.position.y) >= CONTROL_N
    )

  def update(self, active, CS, VM, params, steer_limited_by_safety,
             desired_curvature, calibrated_pose, curvature_limited, lat_delay):
    pid_log = log.ControlsState.LateralTorqueState.new_message()
    pid_log.version = VERSION

    if not active:
      self._curv_rate_filter.x = 0.0
      self._prev_desired_curvature = 0.0
      self._first_active_frame = True
      return 0.0, 0.0, pid_log

    # ─── 1. PHYSICS FEEDFORWARD ──────────────────────────────────────
    # Primary torque: curvature → lateral acceleration → torque
    ff_lat_accel = desired_curvature * CS.vEgo ** 2
    ff_torque = self._to_torque(ff_lat_accel)

    # ─── 2. STATE FEEDBACK (LQR-style, bounded) ─────────────────────
    fb_torque = 0.0
    if self.model_valid:
      idx = min(self._lookahead_idx, len(self.model_v2.position.y) - 1)

      # Lateral error: modelV2 position.y = predicted lane center offset
      # Positive = lane is to the left → steer left (positive torque)
      lat_error = float(self.model_v2.position.y[idx])

      # Heading error: modelV2 orientation.z = predicted heading offset
      # Positive = road curves left → steer left (positive torque)
      heading_error = float(self.model_v2.orientation.z[idx])

      # Guard against NaN/Inf from model glitches
      if not (math.isfinite(lat_error) and math.isfinite(heading_error)):
        lat_error = 0.0
        heading_error = 0.0

      # Speed-dependent gains
      K_lat = float(np.interp(CS.vEgo, LQR_K_LAT_BP, LQR_K_LAT_V))
      K_psi = float(np.interp(CS.vEgo, LQR_K_PSI_BP, LQR_K_PSI_V))

      # Feedback lateral acceleration correction (bounded)
      fb_lat_accel = (
        K_lat * lat_error
        + K_psi * heading_error * CS.vEgo
      )
      fb_lat_accel = float(np.clip(fb_lat_accel, -LQR_FB_LAT_LIMIT, LQR_FB_LAT_LIMIT))

      fb_torque = self._to_torque(fb_lat_accel)

    # ─── 3. CURVATURE RATE LEAD COMPENSATION ────────────────────────
    if self._first_active_frame:
      # First frame: skip derivative to avoid spike
      self._prev_desired_curvature = desired_curvature
      self._curv_rate_filter.x = 0.0
      self._first_active_frame = False
      curv_rate_torque = 0.0
    else:
      raw_curv_rate = (desired_curvature - self._prev_desired_curvature) / self.dt
      self._prev_desired_curvature = desired_curvature
      curv_rate = self._curv_rate_filter.update(raw_curv_rate)

      # Curvature increasing → lead torque (anticipate curve entry)
      # Curvature decreasing → opposing torque (anticipate curve exit)
      curv_rate_lat_accel = LQR_CURV_RATE_GAIN * curv_rate * CS.vEgo ** 2
      curv_rate_torque = self._to_torque(curv_rate_lat_accel)

    # ─── 4. ROLL COMPENSATION ───────────────────────────────────────
    roll_compensation = params.roll * ACCELERATION_DUE_TO_GRAVITY
    roll_torque = self._to_torque(
      -roll_compensation - self.torque_params.latAccelOffset
    )

    # ─── 5. FRICTION COMPENSATION ───────────────────────────────────
    measured_curvature = -VM.calc_curvature(
      math.radians(CS.steeringAngleDeg - params.angleOffsetDeg),
      CS.vEgo, params.roll)
    measured_lat_accel = measured_curvature * CS.vEgo ** 2

    lat_accel_error = ff_lat_accel - measured_lat_accel
    if not self.model_valid:
      lat_accel_error = 0.0

    steering_angle_deadzone_deg = self.torque_params.steeringAngleDeadzoneDeg
    curvature_deadzone = abs(VM.calc_curvature(
      math.radians(steering_angle_deadzone_deg), CS.vEgo, 0.0))
    lateral_accel_deadzone = curvature_deadzone * CS.vEgo ** 2

    friction_torque = self._to_torque(get_friction(
      lat_accel_error, lateral_accel_deadzone,
      FRICTION_THRESHOLD, self.torque_params))

    # ─── 6. TOTAL OUTPUT ────────────────────────────────────────────
    output_torque = ff_torque + fb_torque + curv_rate_torque + roll_torque + friction_torque

    # Deadband crossing: torque below EPS threshold won't move the wheel.
    # Boost to minimum to ensure any steering intent actually reaches the rack.
    # During direction changes, allow zero-crossing to prevent oscillation.
    min_torque = self.torque_params.friction * DEADBAND_FACTOR
    if output_torque != 0.0 and abs(output_torque) < min_torque:
      if self._prev_output_torque == 0.0 or (output_torque > 0) == (self._prev_output_torque > 0):
        output_torque = math.copysign(min_torque, output_torque)
    self._prev_output_torque = output_torque

    output_torque = float(np.clip(output_torque, -self.steer_max, self.steer_max))

    # ─── LOGGING ────────────────────────────────────────────────────
    pid_log.active = True
    pid_log.p = float(ff_torque)
    pid_log.i = float(fb_torque)            # repurposed: state feedback
    pid_log.d = float(curv_rate_torque)     # repurposed: curvature rate lead
    pid_log.f = float(friction_torque)
    pid_log.output = float(-output_torque)
    pid_log.actualLateralAccel = float(measured_lat_accel)
    pid_log.desiredLateralAccel = float(ff_lat_accel)
    pid_log.saturated = bool(abs(output_torque) >= self.steer_max * 0.99)

    return -output_torque, 0.0, pid_log

  def _to_torque(self, lat_accel: float) -> float:
    """Convert lateral acceleration (m/s²) to steering torque (Nm)."""
    return self.torque_from_lateral_accel(lat_accel, self.torque_params)
