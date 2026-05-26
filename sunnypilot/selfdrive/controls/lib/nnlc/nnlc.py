"""
Steering Large Model — NNLC integration (hardened).

Full pipeline:
  Features → Two-Point Baseline → SteeringModel → Post-Processor → Output
       │              │                    │                │
       │         always safe         personalization    human-like
       │         fallback when        incremental       shaping
       │         model uncertain      above baseline

Degradation strategy:
  Level 0: Full system (baseline + model + personalization + post-processing)
  Level 1: ModelV2 down → baseline degrades to near-point only
  Level 2: Model unhealthy → fall back to V0 PID
  Level 3: Everything fails → V0 PID pure

Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
"""
from collections import deque
import math
import time
import numpy as np

from opendbc.car.lateral import FRICTION_THRESHOLD, get_friction
from opendbc.sunnypilot.car.interfaces import LatControlInputs
from opendbc.sunnypilot.car.lateral_ext import get_friction as get_friction_in_torque_space
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext_base import LatControlTorqueExtBase, sign

from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.steering_features import SteeringFeatureExtractor
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.steering_model import SteeringLargeModel
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.online_trainer import OnlineTrainer
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.driving_style import DrivingStyleCollector
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.torque_post import TorquePostProcessor
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.two_point_model import TwoPointModel
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.weight_persistence import WeightManager

LOW_SPEED_X = [0, 10, 20, 30]
LOW_SPEED_Y = [1.2, 0.4, 0, 0]


def roll_pitch_adjust(roll, pitch):
  return roll * math.cos(pitch)


class NeuralNetworkLateralControl(LatControlTorqueExtBase):
  """Human-like lateral control with online personalization and full hardening."""

  # ── Degradation levels ──
  DEGRADE_FULL = 0       # Everything working
  DEGRADE_MODELV2 = 1    # ModelV2 down, baseline near-point only
  DEGRADE_MODEL = 2      # Neural model unhealthy, use baseline + V0 PID
  DEGRADE_V0_ONLY = 3    # Total fallback to V0 PID

  def __init__(self, lac_torque, CP, CP_SP, CI):
    super().__init__(lac_torque, CP, CP_SP, CI)
    self.params = Params()
    self.enabled = self.params.get_bool("NeuralNetworkLateralControl")
    self.has_nn_model = True

    # ── Initialize all components safely ──
    try:
      self.feature_extractor = SteeringFeatureExtractor(CP)
      self.steering_model = SteeringLargeModel(input_dim=84)
    except Exception:
      self.has_nn_model = False
      self.feature_extractor = None
      self.steering_model = None

    try:
      self.two_point = TwoPointModel(self.lac_torque.torque_params)
    except Exception:
      self.two_point = None

    try:
      self.post_processor = TorquePostProcessor(
        max_torque=float(lac_torque.steer_max),
        dt=0.01,
      )
    except Exception:
      self.post_processor = None

    try:
      if self.steering_model is not None:
        self.trainer = OnlineTrainer(self.steering_model)
        self.weight_manager = WeightManager(self.steering_model, self.params)
        self.weight_manager.load()
      else:
        self.trainer = None
        self.weight_manager = None
    except Exception:
      self.trainer = None
      self.weight_manager = None

    # ── Driving style collector (always active, disengaged or engaged) ──
    try:
      self.style_collector = DrivingStyleCollector()
    except Exception:
      self.style_collector = None

    # ── Smooth blending ──
    self._model_blend = FirstOrderFilter(0.0, 0.08, 0.01)
    self._model_active = False
    self._degrade_level = self.DEGRADE_FULL if self.steering_model is not None else self.DEGRADE_MODEL
    self._degrade_counter = 0
    self._consecutive_nans = 0

    # Filters
    self._lsf_filter = FirstOrderFilter(0.0, 0.3, 0.01)
    self.pitch = FirstOrderFilter(0.0, 0.5, 0.01)
    self.pitch_last = 0.0

    # Time offsets
    self.future_times = [0.3, 0.6, 1.0, 1.5]
    self.nn_future_times = [i + self.desired_lat_jerk_time for i in self.future_times]
    self.past_times = [-0.3, -0.2, -0.1]
    history_check_frames = [int(abs(i) * 100) for i in self.past_times]
    self.history_frame_offsets = [history_check_frames[0] - i for i in history_check_frames]
    self.lateral_accel_desired_deque = deque(maxlen=history_check_frames[0])
    self.roll_deque = deque(maxlen=history_check_frames[0])
    self.error_deque = deque(maxlen=history_check_frames[0])
    self.past_future_len = len(self.past_times) + len(self.nn_future_times)

    self._v1_torque = 0.0
    self._last_output = 0.0
    self._frame = 0
    self._startup_time = time.monotonic()
    self._last_health_log = 0.0

  @property
  def _nnlc_enabled(self):
    return self.enabled and self.model_valid and self.has_nn_model

  def update_limits(self):
    if not self._nnlc_enabled:
      return
    try:
      self._pid.set_limits(self.lac_torque.steer_max, -self.lac_torque.steer_max)
    except Exception:
      pass

  def update_lateral_lag(self, lag):
    super().update_lateral_lag(lag)
    self.nn_future_times = [t + self.desired_lat_jerk_time for t in self.future_times]

  # ── V0 baseline (always available) ──
  def _compute_v0_torque(self, CS):
    """Compute V0 PID output as safety baseline."""
    try:
      self.update_feedforward_torque_space(CS)
      freeze = self._steer_limited_by_safety or CS.steeringPressed or CS.vEgo < 2.0
      self._v1_torque = self._pid.update(
        self._pid_log.error, feedforward=self._ff, speed=CS.vEgo, freeze_integrator=freeze)
    except Exception:
      self._v1_torque = 0.0

  def update_feedforward_torque_space(self, CS):
    """V0-compatible FF computation."""
    try:
      torque_from_setpoint = self.torque_from_lateral_accel_in_torque_space(
        LatControlInputs(self._setpoint, self._roll_compensation, CS.vEgo, CS.aEgo),
        self.lac_torque.torque_params, gravity_adjusted=False)
      torque_from_measurement = self.torque_from_lateral_accel_in_torque_space(
        LatControlInputs(self._measurement, self._roll_compensation, CS.vEgo, CS.aEgo),
        self.lac_torque.torque_params, gravity_adjusted=False)
      self._pid_log.error = float(torque_from_setpoint - torque_from_measurement)
      self._ff = self.torque_from_lateral_accel_in_torque_space(
        LatControlInputs(self._gravity_adjusted_lateral_accel, self._roll_compensation, CS.vEgo, CS.aEgo),
        self.lac_torque.torque_params, gravity_adjusted=True)
      self._ff += get_friction_in_torque_space(
        self._desired_lateral_accel - self._actual_lateral_accel,
        self._lateral_accel_deadzone, FRICTION_THRESHOLD, self.lac_torque.torque_params)
    except Exception:
      pass

  # ── Main update ──
  def update_neural_network_feedforward(self, CS, params, calibrated_pose) -> None:
    if not self._nnlc_enabled:
      return

    self._frame += 1
    v = max(CS.vEgo, 0.5)

    # ── Pre-checks ──
    if v < 0.3 or self.steering_model is None:
      self._compute_v0_torque(CS)
      self._output_torque = self._v1_torque
      return

    # ── 1. Setpoint / measurement ──
    try:
      low_speed_factor = float(np.interp(v, LOW_SPEED_X, LOW_SPEED_Y))
      low_speed_factor = self._lsf_filter.update(low_speed_factor)
      self._setpoint = self._desired_lateral_accel + low_speed_factor * self._desired_curvature
      self._measurement = self._actual_lateral_accel + low_speed_factor * self._actual_curvature
    except Exception:
      self._setpoint = self._desired_lateral_accel
      self._measurement = self._actual_lateral_accel

    # ── 2. Roll/pitch ──
    roll = 0.0
    try:
      roll = float(params.roll) if hasattr(params, 'roll') else 0.0
      if calibrated_pose is not None:
        pitch = self.pitch.update(calibrated_pose.orientation.pitch)
        roll = roll_pitch_adjust(roll, pitch)
        self.pitch_last = pitch
    except Exception:
      pass
    self.roll_deque.append(roll)
    self.lateral_accel_desired_deque.append(self._desired_lateral_accel)

    # ── 3. Always compute V0 baseline ──
    self._compute_v0_torque(CS)
    v0_torque = float(self._v1_torque)

    # ── 4. Degradation management ──
    self._update_degradation()

    if self._degrade_level >= self.DEGRADE_MODEL:
      # Neural model unhealthy → V0 PID only
      self._output_torque = self._postprocess(v0_torque, CS, v)
      return

    # ── 5. Extract features ──
    driver_torque = float(getattr(CS, 'steeringTorque', 0.0))
    lateral_error = self._desired_lateral_accel - self._actual_lateral_accel
    heading_error = (self._desired_curvature - self._actual_curvature) * v * 0.2

    try:
      features = self.feature_extractor.extract(
        CS=CS, model_v2=self.model_v2, params=params, live_params=None,
        desired_lateral_accel=self._desired_lateral_accel,
        actual_lateral_accel=self._actual_lateral_accel,
        desired_curvature=self._desired_curvature,
        actual_curvature=self._actual_curvature,
        lateral_error=lateral_error, heading_error=heading_error,
        driver_torque=driver_torque,
      )
    except Exception:
      self._degrade_level = max(self._degrade_level, self.DEGRADE_MODEL)
      self._output_torque = self._postprocess(v0_torque, CS, v)
      return

    # ── 6. Two-point physics baseline ──
    physics_torque = v0_torque
    physics_diag = {}
    if self.two_point is not None and self._degrade_level <= self.DEGRADE_MODELV2:
      try:
        modelv2_ok = self.model_v2 is not None and self.feature_extractor.modelv2_health > 0.5
        far1 = float(np.interp(1.0, ModelConstants.T_IDXS, self.model_v2.position.y)) if modelv2_ok else 0.0
        far2 = float(np.interp(2.0, ModelConstants.T_IDXS, self.model_v2.position.y)) if modelv2_ok else 0.0
        physics_torque, physics_diag = self.two_point.compute(
          lateral_error, self._desired_curvature, self._actual_curvature,
          far1, far2, v, modelv2_ok)
        physics_torque = float(np.clip(physics_torque, -self.lac_torque.steer_max, self.lac_torque.steer_max))
      except Exception:
        pass

    # ── 7. Neural model forward (outputs steering angle degrees) ──
    model_angle_deg = 0.0
    gates = np.ones(4) * 0.25
    try:
      model_angle_deg, gates = self.steering_model.forward(features)
      if np.isnan(model_angle_deg) or np.isinf(model_angle_deg):
        self._consecutive_nans += 1
        model_angle_deg = 0.0
      else:
        self._consecutive_nans = 0
    except Exception:
      self._consecutive_nans += 1
      model_angle_deg = 0.0

    if self._consecutive_nans > 50:
      self._degrade_level = max(self._degrade_level, self.DEGRADE_MODEL)

    # ── Convert steering angle → torque ──
    angle_rad = float(np.radians(model_angle_deg))
    tp = self.lac_torque.torque_params
    model_torque = float(tp.latAccelFactor * angle_rad * v * v / max(tp.latAccelFactor * 10.0, 0.01))
    model_torque = float(np.clip(model_torque, -self.lac_torque.steer_max, self.lac_torque.steer_max))

    # ── 8. Blend model with V0 ──
    model_confidence = float(self.steering_model.personalization_confidence)
    blend_target = 0.05 + 0.65 * model_confidence  # 5% → 70%

    if CS.steeringPressed:
      blend_target *= 0.15
    if self._degrade_level >= self.DEGRADE_MODELV2:
      blend_target *= 0.5

    blend = self._model_blend.update(blend_target)
    self._model_active = blend > 0.03

    # Blend physics baseline, model, and V0
    blended = model_torque * blend + physics_torque * (1.0 - blend) * 0.6 + v0_torque * (1.0 - blend) * 0.4

    # ── 9. Post-processing ──
    self._output_torque = self._postprocess(blended, CS, v, lateral_error)

    # ── 10. Driving style collection (always, regardless of engagement) ──
    try:
      if self.style_collector is not None:
        driver_angle = float(getattr(CS, 'steeringAngleDeg', 0.0))
        driver_torque_eps = float(getattr(CS, 'steeringTorqueEps', 0.0))
        driver_rate = float(getattr(CS, 'steeringRateDeg', 0.0))
        actual_curvature = self._actual_curvature
        lat_error = lateral_error
        is_engaged = getattr(CS, 'enabled', False) if hasattr(CS, 'enabled') else CS.steeringPressed
        self.style_collector.update(
          steering_angle=driver_angle,
          steering_rate=driver_rate,
          steering_torque=driver_torque_eps,
          curvature=actual_curvature,
          lateral_error=lat_error,
          openpilot_engaged=is_engaged,
        )
    except Exception:
      pass

    # ── 11. Online training (style-driven, KL-triggered) ──
    try:
      if self.trainer is not None:
        # Add sample (uses driver angle as target, not torque)
        if CS.steeringPressed:
          driver_angle = float(getattr(CS, 'steeringAngleDeg', 0.0))
          system_angle = float(model_angle_deg)
          self.trainer.add_sample(
            features=features,
            driver_steering_angle=driver_angle,
            system_steering_angle=system_angle,
            v_ego=v,
            curvature=self._actual_curvature,
          )
        # Train when triggered by KL divergence or periodic
        self.trainer.update(style_collector=self.style_collector)
    except Exception:
      pass

    # ── 12. Weight persistence ──
    try:
      if self.weight_manager is not None:
        self.weight_manager.update()
    except Exception:
      pass

    # ── 13. Health logging (every 5 seconds) ──
    now = time.monotonic()
    if now - self._last_health_log > 5.0:
      self._last_health_log = now
      try:
        health = self.steering_model.check_health()
        if not health['healthy']:
          self._degrade_level = max(self._degrade_level, self.DEGRADE_MODEL)
      except Exception:
        pass

    # ── 14. Update PID log ──
    self._pid_log.error = float(lateral_error)
    self._pid_log.active = True
    self._last_output = self._output_torque

  def _postprocess(self, torque, CS, v, lateral_error=0.0):
    """Apply post-processing with safety fallback."""
    if self.post_processor is None:
      return float(np.clip(torque, -self.lac_torque.steer_max, self.lac_torque.steer_max))
    try:
      return self.post_processor.process(
        raw_torque=torque,
        lateral_error=lateral_error,
        actual_lateral_accel=self._actual_lateral_accel,
        desired_curvature=self._desired_curvature,
        actual_curvature=self._actual_curvature,
        v_ego=max(v, 0.5),
        steering_pressed=CS.steeringPressed,
        active=True,
      )
    except Exception:
      return float(np.clip(torque, -self.lac_torque.steer_max, self.lac_torque.steer_max))

  def _update_degradation(self):
    """Monitor system health and adjust degradation level."""
    # Check modelV2 health
    if self.feature_extractor is not None and self.feature_extractor.modelv2_health < 0.3:
      self._degrade_level = max(self._degrade_level, self.DEGRADE_MODELV2)
    elif self.feature_extractor is not None and self.feature_extractor.modelv2_health > 0.7:
      if self._degrade_level == self.DEGRADE_MODELV2:
        self._degrade_level = self.DEGRADE_FULL

    # Check steering model health
    if self.steering_model is not None:
      health = self.steering_model.check_health()
      if not health['healthy'] and self._degrade_level < self.DEGRADE_MODEL:
        self._degrade_counter += 1
        if self._degrade_counter > 300:  # 3 seconds of bad health
          self._degrade_level = max(self._degrade_level, self.DEGRADE_MODEL)
      else:
        self._degrade_counter = max(0, self._degrade_counter - 1)
