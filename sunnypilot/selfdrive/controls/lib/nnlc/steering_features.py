"""
Steering Feature Extractor — hardened.

84-dim features from openpilot data with:
  - NaN/Inf guards on every feature
  - Input value clipping to normalization ranges
  - ModelV2 graceful degradation (partial/missing data)
  - Division-by-zero guards everywhere
  - Distribution shift detection (feature range monitoring)
"""
import numpy as np
from collections import deque


def _safe_float(val, default=0.0, clip_min=-8.0, clip_max=8.0):
  """Return a safe float32, replacing NaN/Inf with default and clipping."""
  try:
    v = float(val)
  except (TypeError, ValueError):
    return float(default)
  if np.isnan(v) or np.isinf(v):
    return float(default)
  return float(np.clip(v, clip_min, clip_max))


def _safe_interp(t, t_idxs, arr):
  """Interpolate modelV2 timeseries safely."""
  if arr is None or len(arr) == 0:
    return 0.0
  if t <= t_idxs[0]:
    return _safe_float(arr[0])
  if t >= t_idxs[-1]:
    return _safe_float(arr[-1])
  return _safe_float(np.interp(t, t_idxs, arr))


class SteeringFeatureExtractor:
  """Extract 84 normalized features with full safety hardening."""

  # Feature dimension constants
  # Feature dimension constants
  N_STATE = 12       # current vehicle state
  N_LATERAL = 8      # lateral control state
  N_TWOPOINT = 6     # two-point visual model
  N_FUTURE = 24      # modelV2 future (8 steps × 3 channels)
  N_PAST = 20        # past trajectory (5 steps × 4 channels)
  N_SCENE = 8        # scene context
  N_VEHICLE = 6      # vehicle static params
  FEATURE_DIM = N_STATE + N_LATERAL + N_TWOPOINT + N_FUTURE + N_PAST + N_SCENE + N_VEHICLE  # 84

  # Clipping bounds per feature group (min, max)
  CLIP = {
    'state': (-5.0, 5.0),
    'lateral': (-4.0, 4.0),
    'two_point': (-3.0, 3.0),
    'future': (-4.0, 4.0),
    'past': (-4.0, 4.0),
    'scene': (0.0, 1.0),      # boolean features
    'vehicle': (-2.0, 2.0),
  }

  def __init__(self, CP):
    # Safely extract vehicle params with defaults
    try:
      self._veh = {
        'latAccelFactor': _safe_float(CP.lateralTuning.torque.latAccelFactor, 2.0),
        'friction': _safe_float(CP.lateralTuning.torque.friction, 0.1),
        'steerRatio': _safe_float(CP.steerRatio, 15.0),
        'wheelbase': _safe_float(CP.wheelbase, 2.7),
        'tireStiffnessFront': _safe_float(CP.tireStiffnessFront, 192150.0, 0.0, 1e7),
        'tireStiffnessRear': _safe_float(CP.tireStiffnessRear, 202500.0, 0.0, 1e7),
        'steerActuatorDelay': _safe_float(CP.steerActuatorDelay, 0.15, 0.0, 1.0),
        'centerToFront': _safe_float(CP.centerToFront, 1.08, 0.5, 3.0),
        'mass': _safe_float(CP.mass, 1300.0, 500.0, 5000.0),
        'rotationalInertia': _safe_float(CP.rotationalInertia, 2327.0, 500.0, 10000.0),
      }
    except Exception:
      self._veh = {k: 1.0 for k in ['latAccelFactor', 'friction', 'steerRatio', 'wheelbase',
                                      'tireStiffnessFront', 'tireStiffnessRear', 'steerActuatorDelay',
                                      'centerToFront', 'mass', 'rotationalInertia']}

    # History ring buffers — pre-allocate with zeros
    maxlen = 200
    self._hist_desired_lat = deque([0.0] * maxlen, maxlen=maxlen)
    self._hist_actual_lat = deque([0.0] * maxlen, maxlen=maxlen)
    self._hist_curvature = deque([0.0] * maxlen, maxlen=maxlen)
    self._hist_roll = deque([0.0] * maxlen, maxlen=maxlen)
    self._hist_steer_torque = deque([0.0] * maxlen, maxlen=maxlen)
    self._prev_far_angles = deque([0.0] * 10, maxlen=10)

    self._frame = 0
    self._last_intervention_frame = -10000

    # ── Distribution shift detection ──
    self._feature_means = np.zeros(self.FEATURE_DIM, dtype=np.float32)
    self._feature_stds = np.zeros(self.FEATURE_DIM, dtype=np.float32)
    self._feature_count = 0
    self._modelv2_valid_frames = 0
    self._modelv2_invalid_frames = 0
    self._cached_t_idxs = None

  def _get_past(self, buf, offset, default=0.0, clip_min=-4.0, clip_max=4.0):
    """Read past value at offset, with bounds check and NaN guard."""
    try:
      if offset < len(buf):
        return _safe_float(buf[-1 - offset], default, clip_min, clip_max)
    except (IndexError, TypeError):
      pass
    return _safe_float(default, default, clip_min, clip_max)

  @property
  def feature_dim(self):
    return self.FEATURE_DIM

  @property
  def modelv2_health(self):
    """Fraction of frames where modelV2 was valid."""
    total = self._modelv2_valid_frames + self._modelv2_invalid_frames
    return self._modelv2_valid_frames / max(total, 1)

  def extract(self, CS, model_v2, params, live_params, desired_lateral_accel,
              actual_lateral_accel, desired_curvature, actual_curvature,
              lateral_error, heading_error, driver_torque):
    """Build 84-dim feature vector with full safety hardening."""
    f = np.zeros(self.FEATURE_DIM, dtype=np.float32)
    v = max(_safe_float(CS.vEgo, 0.5, 0.0, 60.0), 0.5)

    # Track modelV2 health
    is_model_valid = model_v2 is not None
    if is_model_valid:
      self._modelv2_valid_frames += 1
      # Cache T_IDXS for interpolation
      if self._cached_t_idxs is None:
        try:
          from openpilot.selfdrive.modeld.constants import ModelConstants
          self._cached_t_idxs = ModelConstants.T_IDXS
        except ImportError:
          self._cached_t_idxs = np.linspace(0, 10, 33)
    else:
      self._modelv2_invalid_frames += 1

    # ── Category 1: Current State (12) ──
    idx = 0
    f[idx] = _safe_float(v / 40.0); idx += 1                          # 0
    f[idx] = _safe_float(min(v * v / 1600.0, 2.0)); idx += 1          # 1
    f[idx] = _safe_float(CS.aEgo / 5.0); idx += 1                     # 2
    f[idx] = _safe_float(CS.steeringAngleDeg / 500.0); idx += 1       # 3
    f[idx] = _safe_float(CS.steeringRateDeg / 100.0); idx += 1        # 4
    f[idx] = _safe_float(driver_torque / 5.0); idx += 1               # 5
    f[idx] = 1.0 if CS.steeringPressed else 0.0; idx += 1             # 6
    f[idx] = _safe_float(getattr(params, 'roll', 0.0) / 0.15); idx += 1  # 7
    f[idx] = _safe_float(getattr(params, 'pitch', 0.0) / 0.15); idx += 1 # 8
    f[idx] = _safe_float(getattr(CS, 'yawRate', 0.0) / 0.5); idx += 1    # 9
    f[idx] = 1.0 if getattr(CS, 'gasPressed', False) else 0.0; idx += 1  # 10
    f[idx] = _safe_float(getattr(CS, 'brakePressed', 0) / 1.0); idx += 1 # 11

    # ── Category 2: Lateral State (8) ──
    f[idx] = _safe_float(desired_lateral_accel / 5.0); idx += 1       # 12
    f[idx] = _safe_float(actual_lateral_accel / 5.0); idx += 1        # 13
    f[idx] = _safe_float((desired_lateral_accel - actual_lateral_accel) / 3.0); idx += 1  # 14
    f[idx] = _safe_float(desired_curvature / 0.1); idx += 1           # 15
    f[idx] = _safe_float(actual_curvature / 0.1); idx += 1            # 16
    f[idx] = _safe_float((desired_curvature - actual_curvature) / 0.1); idx += 1  # 17
    # Lateral jerk (safe computation)
    lat_jerk = 0.0
    if len(self._hist_desired_lat) >= 2:
      raw_jerk = self._hist_desired_lat[-1] - self._hist_desired_lat[-2]
      lat_jerk = _safe_float(raw_jerk / 0.01, 0.0, -10.0, 10.0)
    f[idx] = _safe_float(lat_jerk / 3.0); idx += 1                    # 18
    f[idx] = _safe_float(heading_error / 0.3); idx += 1               # 19

    # ── Category 3: Two-Point Model Features (6) ──
    # These directly encode the Salvucci & Gray visual angles
    near_angle = _safe_float(lateral_error / 8.0)
    f[idx] = near_angle; idx += 1                                     # 20

    # Far point angles
    far1 = _safe_interp(1.0, self._cached_t_idxs or [0], 
                        model_v2.position.y if is_model_valid else None)
    far2 = _safe_interp(2.0, self._cached_t_idxs or [0],
                        model_v2.position.y if is_model_valid else None)
    far_angle_1 = _safe_float(far1 / max(v * 1.0, 1.0), 0.0, -2.0, 2.0)
    far_angle_2 = _safe_float(far2 / max(v * 2.0, 1.0), 0.0, -2.0, 2.0)
    f[idx] = far_angle_1 / 0.3; idx += 1                              # 21
    f[idx] = far_angle_2 / 0.3; idx += 1                              # 22

    # Far angle derivatives
    self._prev_far_angles.append(far_angle_1)
    far_rate_1 = _safe_float(
      (self._prev_far_angles[-1] - self._prev_far_angles[-2]) / 0.01
      if len(self._prev_far_angles) >= 2 else 0.0)
    far_rate_2 = _safe_float(
      (self._prev_far_angles[-1] - self._prev_far_angles[-5]) / 0.04
      if len(self._prev_far_angles) >= 5 else 0.0)
    f[idx] = _safe_float(far_rate_1 / 0.5); idx += 1                  # 23
    f[idx] = _safe_float(far_rate_2 / 0.5); idx += 1                  # 24
    f[idx] = _safe_float((far_angle_1 - near_angle) / 0.5); idx += 1  # 25

    # ── Category 4: Future Trajectory from modelV2 (24) ──
    future_ts = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]
    if is_model_valid and self._cached_t_idxs is not None:
      for t in future_ts:
        lat_acc = _safe_interp(t, self._cached_t_idxs, model_v2.acceleration.y)
        lat_pos = _safe_interp(t, self._cached_t_idxs, model_v2.position.y)
        heading = _safe_interp(t, self._cached_t_idxs, model_v2.orientation.z)
        f[idx] = _safe_float(lat_acc / 3.0); idx += 1
        f[idx] = _safe_float(lat_pos / 2.0); idx += 1
        f[idx] = _safe_float(heading / 0.3); idx += 1
    else:
      # ModelV2 unavailable — zero-fill
      idx += 24

    # ── Category 5: Past Trajectory (20) ──
    past_offsets = [2, 5, 10, 20, 50]
    for off in past_offsets:
      f[idx] = self._get_past(self._hist_desired_lat, off) / 5.0; idx += 1
      f[idx] = self._get_past(self._hist_actual_lat, off) / 5.0; idx += 1
      f[idx] = self._get_past(self._hist_curvature, off) / 0.1; idx += 1
      f[idx] = self._get_past(self._hist_roll, off) / 0.15; idx += 1

    # ── Category 6: Scene Context (8) ──
    des_k_abs = abs(_safe_float(desired_curvature))
    act_k_abs = abs(_safe_float(actual_curvature))
    is_straight = des_k_abs < 0.0005 and act_k_abs < 0.0005
    is_curve_entry = des_k_abs > act_k_abs * 1.3 and des_k_abs > 0.001
    is_curve_exit = des_k_abs < act_k_abs * 0.5 and act_k_abs > 0.001

    f[idx] = 1.0 if is_curve_entry else 0.0; idx += 1                # 70
    f[idx] = 1.0 if is_curve_exit else 0.0; idx += 1                 # 71
    f[idx] = 1.0 if is_straight else 0.0; idx += 1                   # 72
    k_class = 0
    if des_k_abs > 0.02: k_class = 3
    elif des_k_abs > 0.01: k_class = 2
    elif des_k_abs > 0.003: k_class = 1
    f[idx] = _safe_float(k_class / 3.0); idx += 1                    # 73
    frames_since = max(0, self._frame - self._last_intervention_frame)
    f[idx] = _safe_float(min(frames_since / 6000.0, 1.0)); idx += 1  # 74
    f[idx] = 1.0 if CS.steeringPressed else 0.0; idx += 1            # 75
    f[idx] = 0.0; idx += 1                                            # 76 reserved
    f[idx] = 0.0; idx += 1                                            # 77 reserved

    # ── Category 7: Vehicle Parameters (6) ──
    f[idx] = _safe_float(self._veh['latAccelFactor'] / 5.0); idx += 1    # 78
    f[idx] = _safe_float(self._veh['friction'] / 0.5); idx += 1          # 79
    f[idx] = _safe_float(self._veh['steerRatio'] / 20.0); idx += 1       # 80
    f[idx] = _safe_float(self._veh['wheelbase'] / 3.0); idx += 1         # 81
    f[idx] = _safe_float(self._veh['tireStiffnessFront'] / 300000.0); idx += 1  # 82
    f[idx] = _safe_float(self._veh['centerToFront'] / 2.0); idx += 1     # 83

    # ── Final NaN sweep ──
    f = np.nan_to_num(f, nan=0.0, posinf=4.0, neginf=-4.0).astype(np.float32)

    # ── Update distribution tracking (EMA) ──
    self._feature_count += 1
    if self._feature_count > 1:
      alpha = 0.001
      self._feature_means = (1 - alpha) * self._feature_means + alpha * f
      self._feature_stds = (1 - alpha) * self._feature_stds + alpha * np.abs(f - self._feature_means)

    # ── Update history buffers ──
    self._hist_desired_lat.append(_safe_float(desired_lateral_accel))
    self._hist_actual_lat.append(_safe_float(actual_lateral_accel))
    self._hist_curvature.append(_safe_float(actual_curvature))
    self._hist_roll.append(_safe_float(getattr(params, 'roll', 0.0)))
    self._hist_steer_torque.append(_safe_float(driver_torque))
    self._frame += 1

    if CS.steeringPressed:
      self._last_intervention_frame = self._frame

    return f

  def check_distribution_shift(self, features, threshold=3.0):
    """Check if current features are out-of-distribution (returns anomaly score)."""
    if self._feature_count < 100:
      return 0.0
    z_scores = np.abs(features - self._feature_means) / np.maximum(self._feature_stds, 0.01)
    return float(np.mean(z_scores > threshold))
