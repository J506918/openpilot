"""
Two-Point Model — physics baseline, hardened.

Always runs. Provides safe torque baseline.
When modelV2 is unavailable → degrades to near-point-only (still safe).
"""
import numpy as np


class TwoPointModel:
  """Physics human steering model. Safe by construction."""

  GAIN_SPEEDS = [2.0, 8.0, 15.0, 30.0]
  NEAR_GAINS = [0.20, 0.45, 0.65, 0.90]
  FAR_GAINS = [2.0, 1.6, 1.2, 0.9]
  FAR_DERIV_GAINS = [0.80, 0.55, 0.35, 0.25]

  def __init__(self, torque_params, dt=0.01):
    self.tp = torque_params
    self.dt = dt
    self.lookahead_near = 8.0
    self.lookahead_far_t = 1.5

    self._prev_far_angle = 0.0
    self._prev_far_angle_2 = 0.0
    self._frame = 0
    self._modelv2_degraded = False
    self._degraded_frames = 0

  def compute(self, lateral_error, desired_curvature, actual_curvature,
              future_lateral_1s, future_lateral_2s, v_ego, modelv2_valid):
    """
    Returns (torque_nm, diagnostics_dict).
    Degrades gracefully when modelV2 is down.
    """
    v = max(v_ego, 1.0)

    # ── Track degradation ──
    if not modelv2_valid:
      self._degraded_frames += 1
      self._modelv2_degraded = self._degraded_frames > 50
    else:
      self._degraded_frames = max(0, self._degraded_frames - 1)
      if self._degraded_frames == 0:
        self._modelv2_degraded = False

    # ── Gains ──
    k_n = float(np.interp(v, self.GAIN_SPEEDS, self.NEAR_GAINS))
    k_f = float(np.interp(v, self.GAIN_SPEEDS, self.FAR_GAINS))
    k_fd = float(np.interp(v, self.GAIN_SPEEDS, self.FAR_DERIV_GAINS))

    # ── Near point (always available) ──
    theta_near = lateral_error / self.lookahead_near

    # ── Far point (modelV2 required) ──
    if self._modelv2_degraded:
      # Degraded mode: use near point only, increase its gain
      theta_far = 0.0
      theta_far_rate = 0.0
      k_n *= 1.5  # compensate for missing preview
    else:
      far_pos = (future_lateral_1s + future_lateral_2s) * 0.5
      lookahead_far = max(v * self.lookahead_far_t, 1.0)
      theta_far = far_pos / lookahead_far

      theta_far_rate = 0.0
      if self._frame >= 5:
        theta_far_rate = (theta_far - self._prev_far_angle_2) / (5 * self.dt)
      elif self._frame >= 1:
        theta_far_rate = (theta_far - self._prev_far_angle) / self.dt

      self._prev_far_angle_2 = self._prev_far_angle
      self._prev_far_angle = theta_far

    self._frame += 1

    # ── Steering angle → torque ──
    delta_desired = k_n * theta_near + k_f * theta_far + k_fd * theta_far_rate
    desired_lat_accel = delta_desired * v * v / max(self.tp.latAccelFactor, 0.01)

    torque = self.tp.latAccelFactor * desired_lat_accel

    # Friction (deadzone-aware)
    if abs(desired_lat_accel) > 0.05:
      torque += np.sign(desired_lat_accel) * self.tp.friction * 0.4

    # Curvature error correction (safety net for understeer)
    curvature_error = desired_curvature - actual_curvature
    torque += curvature_error * v * v * self.tp.latAccelFactor * 0.2

    torque = float(np.clip(torque, -4.0, 4.0))

    return torque, {
      'theta_near': float(theta_near),
      'theta_far': float(theta_far),
      'theta_far_rate': float(theta_far_rate),
      'k_n': k_n, 'k_f': k_f, 'k_fd': k_fd,
      'degraded': self._modelv2_degraded,
    }
