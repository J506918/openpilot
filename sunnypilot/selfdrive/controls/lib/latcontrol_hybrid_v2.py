"""
Hybrid Lateral Control V2 — MPC-Optimized Torque Control
Copyright (c) 2026, sunnypilot contributors

Architecture:
  Full model trajectory → acados MPC solver → optimal curvature → direct torque

Uses all 33 model trajectory points as reference for MPC optimization over
a 10-second horizon with constraints on yaw angle (≤90°) and yaw rate (≤50°/s).

The solver output is a full optimal state trajectory. We extract the first
step's yaw rate as the desired curvature command.

This replaces the previous Hybrid V1 (PID-blending) and V2 (multi-point
feedforward) approaches with a proper full-horizon optimization.
"""

import math
import numpy as np

from cereal import log
from opendbc.car.lateral import get_friction
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.controls.lib.drive_helpers import CAR_ROTATION_RADIUS
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.selfdrive.controls.lib.lateral_mpc_lib.lat_mpc import LateralMpc
from openpilot.selfdrive.controls.lib.lateral_mpc_lib.lat_mpc import N as MPC_N
from openpilot.selfdrive.modeld.constants import ModelConstants

# ─── Tuning Constants ─────────────────────────────────────────────────────
MPC_PATH_WEIGHT = 1.0
MPC_HEADING_WEIGHT = 0.5
MPC_LAT_ACCEL_WEIGHT = 0.1
MPC_LAT_JERK_WEIGHT = 0.05
MPC_STEERING_RATE_WEIGHT = 500

FRICTION_THRESHOLD = 0.3
CURVATURE_LP_FC = 2.0
SPEED_OFFSET = 10.0

VERSION = 2


class HybridLateralControlV2(LatControl):
  """MPC-driven lateral control: full-horizon trajectory optimization."""

  def __init__(self, CP, CP_SP, CI, dt):
    super().__init__(CP, CP_SP, CI, dt)

    # Act as own extension for controlsd's extension.update_model_v2() call
    self.extension = self

    self.torque_params = CP.lateralTuning.torque.as_builder()
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.lateral_accel_from_torque = CI.lateral_accel_from_torque()

    # Model state
    self.model_v2 = None
    self.model_valid = False

    # MPC solver
    self.mpc = LateralMpc()
    self._set_mpc_weights()

    # Low-pass filter for curvature (MPC can produce frame-to-frame noise)
    self.curvature_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * CURVATURE_LP_FC), dt)

    # Pre-allocated arrays for MPC reference trajectory
    self._y_pts = np.zeros(MPC_N + 1)
    self._heading_pts = np.zeros(MPC_N + 1)
    self._yaw_rate_pts = np.zeros(MPC_N + 1)
    self._p = np.zeros((MPC_N + 1, 2))

  def _set_mpc_weights(self):
    """Configure MPC cost weights for lane-keeping behavior."""
    self.mpc.set_weights(
      MPC_PATH_WEIGHT,
      MPC_HEADING_WEIGHT,
      MPC_LAT_ACCEL_WEIGHT,
      MPC_LAT_JERK_WEIGHT,
      MPC_STEERING_RATE_WEIGHT,
    )

  def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction):
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction

  def update_limits(self):
    pass  # Constraints handled by MPC internally

  def update_lateral_lag(self, lag):
    pass

  def update_model_v2(self, model_v2):
    self.model_v2 = model_v2
    self.model_valid = self.model_v2 is not None and len(self.model_v2.position.y) >= 16

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature,
             calibrated_pose, curvature_limited, lat_delay):
    """Main control update — MPC-optimized trajectory + torque conversion."""
    pid_log = log.ControlsState.LateralTorqueState.new_message()
    pid_log.version = VERSION

    if not active:
      self.mpc.reset()
      self.curvature_filter.x = 0.0
      return 0.0, 0.0, pid_log

    # ─── Build MPC reference from model predictions ────────────────────
    if self.model_valid:
      n = min(int(len(self.model_v2.position.y)), MPC_N + 1)
      for i in range(n):
        self._y_pts[i] = self.model_v2.position.y[i]
        self._heading_pts[i] = self.model_v2.orientation.z[i]
        self._yaw_rate_pts[i] = self.model_v2.orientationRate.z[i]
      if n < MPC_N + 1:
        self._y_pts[n:] = self._y_pts[n - 1]
        self._heading_pts[n:] = self._heading_pts[n - 1]
        self._yaw_rate_pts[n:] = self._yaw_rate_pts[n - 1]

      measured_curvature = -VM.calc_curvature(
        math.radians(CS.steeringAngleDeg - params.angleOffsetDeg),
        CS.vEgo, params.roll)
      x0 = np.array([0.0, 0.0, 0.0, measured_curvature * CS.vEgo])

      self._p[:, 0] = CS.vEgo
      self._p[:, 1] = CAR_ROTATION_RADIUS

      self.mpc.run(x0, self._p, self._y_pts, self._heading_pts, self._yaw_rate_pts)

      if self.mpc.solution_status == 0:
        mpc_psi_rate = self.mpc.x_sol[0, 3]
        mpc_curvature = mpc_psi_rate / max(CS.vEgo, 0.5)
        model_curvature = self.curvature_filter.update(mpc_curvature)
      else:
        model_curvature = self.curvature_filter.update(0.0)
    else:
      model_curvature = self.curvature_filter.update(0.0)

    # ─── Torque from curvature ─────────────────────────────────────────
    predicted_lat_accel = model_curvature * CS.vEgo ** 2
    base_torque = self.torque_from_lateral_accel(predicted_lat_accel, self.torque_params)

    # ─── Roll compensation ─────────────────────────────────────────────
    roll_compensation = params.roll * ACCELERATION_DUE_TO_GRAVITY
    roll_torque = self.torque_from_lateral_accel(
      -roll_compensation - self.torque_params.latAccelOffset, self.torque_params)

    # ─── Friction compensation ─────────────────────────────────────────
    measured_curvature = -VM.calc_curvature(
      math.radians(CS.steeringAngleDeg - params.angleOffsetDeg),
      CS.vEgo, params.roll)
    measured_lat_accel = measured_curvature * CS.vEgo ** 2

    lat_accel_error = predicted_lat_accel - measured_lat_accel
    if not self.model_valid or self.mpc.solution_status != 0:
      lat_accel_error = 0.0  # Don't fight the car when MPC is unavailable

    steering_angle_deadzone_deg = self.torque_params.steeringAngleDeadzoneDeg
    curvature_deadzone = abs(VM.calc_curvature(
      math.radians(steering_angle_deadzone_deg), CS.vEgo, 0.0))
    lateral_accel_deadzone = curvature_deadzone * CS.vEgo ** 2

    friction_torque = get_friction(
      lat_accel_error, lateral_accel_deadzone,
      FRICTION_THRESHOLD, self.torque_params)

    # ─── Final output ──────────────────────────────────────────────────
    output_torque = base_torque + friction_torque + roll_torque

    if steer_limited_by_safety:
      output_torque = 0.0
    else:
      output_torque = np.clip(output_torque, -self.steer_max, self.steer_max)

    # ─── Logging ───────────────────────────────────────────────────────
    pid_log.active = True
    pid_log.p = float(base_torque)
    pid_log.i = 0.0
    pid_log.f = float(friction_torque)
    pid_log.output = float(-output_torque)
    pid_log.actualLateralAccel = float(measured_lat_accel)
    pid_log.desiredLateralAccel = float(predicted_lat_accel)
    pid_log.saturated = bool(abs(output_torque) >= self.steer_max * 0.99)

    return -output_torque, 0.0, pid_log
