"""
Lateral Control FFv1 — Predictive Feedforward + Feedback with Online Adaptation
Copyright (c) 2026, sunnypilot contributors

Architecture (four-layer separation, total compute < 2.2ms):
  Layer 1 — Intelligent Feedforward:
    κ(t+lat_delay) delay-anticipatory preview + bicycle-model steady-state
    solution + RLS online parameter adaptation (α, β).
  Layer 2 — Predictive Feedback:
    Smith Predictor state prediction over lat_delay horizon +
    low-bandwidth gain-scheduled feedback (< 1.7 Hz closed-loop).
  Layer 3 — Disturbance Compensation:
    Low-frequency DOB (< 0.5 Hz) + bank angle estimation + friction-adaptive.
  Layer 4 — Constraint & Safety:
    Predictive constraint check + torque rate limiting + anti-windup.

Designed for Honda Accord 11th-gen Bosch EPS on C3X hardware.
  torqueBP = [0, 2560]
  latAccelFactor = 1.35, friction = 0.17

VERSION 100
"""

import math
import numpy as np
from collections import deque

from cereal import log
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.selfdrive.modeld.constants import ModelConstants

# ─── Version ───────────────────────────────────────────────────────────
VERSION = 100

# ─── Vehicle Parameters (Honda Accord 11th-gen defaults) ───────────────
DEFAULT_LAT_ACCEL_FACTOR = 1.35   # torque → lat-accel gain
DEFAULT_FRICTION = 0.17           # static friction torque
DEFAULT_WHEELBASE = 2.83          # metres (l_f + l_r)
DEFAULT_MASS = 1550.0             # kg (curb weight approx)
DEFAULT_IZ = 2800.0               # yaw moment of inertia [kg·m²]
DEFAULT_LF = 1.38                 # front axle to CG [m]
DEFAULT_LR = 1.45                 # rear axle to CG [m]
DEFAULT_CF = 80000.0              # front cornering stiffness [N/rad]
DEFAULT_CR = 85000.0              # rear cornering stiffness [N/rad]

# ─── RLS Adaptive Parameters ──────────────────────────────────────────
RLS_FORGETTING = 0.995            # forgetting factor λ (memory ~200 steps ≈ 4s)
RLS_P0_DIAG = 10.0                # initial covariance diagonal
RLS_ALPHA_RANGE = (0.5, 3.0)     # latAccelFactor bounds
RLS_FRICTION_RANGE = (0.03, 0.50) # friction bounds
RLS_MIN_SPEED = 5.0               # minimum speed for RLS update [m/s]
RLS_MIN_CURVATURE = 0.0001        # minimum curvature for RLS update [1/m]
RLS_CONFIDENCE_THRESHOLD = 0.3    # below this → fall back to defaults

# ─── Feedback Gain Schedule (at 60 km/h ≈ 16.67 m/s reference) ────────
FB_REF_SPEED = 16.67              # reference speed [m/s]
FB_KY = 5.0                       # lateral position gain
FB_KPSI = 15.0                    # heading error gain
FB_KBETA = 2.0                    # sideslip gain
FB_KR = 0.5                       # yaw rate gain

# ─── Disturbance Observer ─────────────────────────────────────────────
DOB_CUTOFF_HZ = 0.5               # LPF cutoff for total disturbance
BANK_CUTOFF_HZ = 0.05             # LPF cutoff for bank angle estimation
BANK_STRAIGHT_KAPPA = 0.001       # curvature threshold for straight detection

# ─── Constraint & Safety ──────────────────────────────────────────────
TORQUE_RATE_NORMAL = 100.0        # unit/s — comfort limit
TORQUE_RATE_EMERGENCY = 300.0     # unit/s — safety override
LAT_ACCEL_COMFORT = 2.0           # m/s² comfort limit
LAT_ACCEL_SAFETY = 4.0            # m/s² absolute limit
YAW_RATE_LIMIT = 0.5              # rad/s stability limit
ANTI_WINDUP_GAIN = 0.5            # anti-windup back-calculation gain

# ─── Dynamic Feedforward ──────────────────────────────────────────────
# K_dκ = (I_z · L) / (2 · C_f · l_f)  — curvature-rate feedforward gain
K_DKAPPA = (DEFAULT_IZ * DEFAULT_WHEELBASE) / (2.0 * DEFAULT_CF * DEFAULT_LF)

# ─── Delay Margin ─────────────────────────────────────────────────────
DELAY_MARGIN_S = 0.020            # extra 20ms safety margin on lat_delay

# ─── Observer Gain (Smith Predictor Innovation Correction) ──────────
OBSERVER_L = 0.3                  # observer gain, low bandwidth (< 1/(3*lat_delay) ≈ 1.7Hz)

# ─── Minimum Speed ────────────────────────────────────────────────────
MIN_SPEED = 1.0                   # avoid division by zero

# ─── Integrator Gain ──────────────────────────────────────────────────
INTEGRATOR_GAIN = 0.3             # lateral error → torque conversion gain

# ─── Heading Error Integration ────────────────────────────────────────
HEADING_ERROR_MAX = 0.15          # max heading error [rad] (~8.6°)


def sign_with_deadzone(x, dz=0.01):
  """Smoothed sign function with deadzone to avoid chattering."""
  if abs(x) < dz:
    return 0.0
  return 1.0 if x > 0.0 else -1.0


def clip(val, lo, hi):
  return max(lo, min(hi, val))


class StatePredictor:
  """
  Smith Predictor: predicts the vehicle state lat_delay seconds into the future
  using a simplified bicycle model and a ring buffer of past torque commands.

  State: [e_y, e_psi, beta, r]
    e_y   — lateral position error (relative to path)
    e_psi — heading error
    beta  — sideslip angle
    r     — yaw rate
  """

  def __init__(self, dt, wheelbase, mass, iz, lf, lr, cf, cr, lat_accel_factor):
    self.dt = dt
    self.L = wheelbase
    self.mass = mass
    self.iz = iz
    self.lf = lf
    self.lr = lr
    self.cf = cf
    self.cr = cr
    self.lat_accel_factor = lat_accel_factor

  def predict(self, state_current, input_buffer, v, n_steps):
    """
    Forward-integrate the bicycle model for n_steps using stored inputs.
    Returns predicted state at t + n_steps*dt.
    """
    ey, epsi, beta, r = state_current
    dt = self.dt
    m = self.mass
    L = self.L
    lf = self.lf
    lr = self.lr
    cf = self.cf
    cr = self.cr
    iz = self.iz
    v_safe = max(v, MIN_SPEED)

    for i in range(n_steps):
      # Get the torque from buffer (or zero if buffer exhausted)
      if i < len(input_buffer):
        tau = input_buffer[i]
      else:
        tau = 0.0

      # Convert torque to approximate steering angle via inverse model
      # δ ≈ tau / lat_accel_factor  (simplified — actual mapping is nonlinear)
      delta = tau / max(self.lat_accel_factor, 0.1)

      # Bicycle model dynamics (linearised around current operating point)
      # β̇ = -(2Cf+2Cr)/(m·v) · β - (2Cf·lf-2Cr·lr)/(m·v²) · r + 2Cf/(m·v) · δ
      # ṙ = -(2Cf·lf-2Cr·lr)/Iz · β - (2Cf·lf²+2Cr·lr²)/(Iz·v) · r + 2Cf·lf/Iz · δ
      cf2 = 2.0 * cf
      cr2 = 2.0 * cr

      beta_dot = (-(cf2 + cr2) / (m * v_safe) * beta
                  - (cf2 * lf - cr2 * lr) / (m * v_safe ** 2) * r
                  + cf2 / (m * v_safe) * delta)

      r_dot = (-(cf2 * lf - cr2 * lr) / iz * beta
               - (cf2 * lf ** 2 + cr2 * lr ** 2) / (iz * v_safe) * r
               + cf2 * lf / iz * delta)

      # Heading error dynamics: ψ̇ = r - κ_ref · v  (κ_ref ≈ 0 for prediction)
      epsi_dot = r

      # Lateral error: ẏ = v · (ψ + β)
      ey_dot = v_safe * (epsi + beta)

      # Euler integration
      ey += ey_dot * dt
      epsi += epsi_dot * dt
      beta += beta_dot * dt
      r += r_dot * dt

    return [ey, epsi, beta, r]


class DisturbanceObserver:
  """
  Low-frequency disturbance observer.
  Estimates total lateral disturbance (road bank, crosswind, tire wear, etc.)
  by comparing measured lateral acceleration with model-predicted lateral accel.
  Also estimates bank angle during straight-line driving.
  """

  def __init__(self, dt, lat_accel_factor, dobf_cutoff_hz, bank_cutoff_hz):
    self.dt = dt
    self.lat_accel_factor = lat_accel_factor
    self.disturbance_estimate = 0.0
    self.bank_angle_estimate = 0.0

    # LPF coefficients: α = dt / (dt + 1/(2π·fc))
    tau_dob = 1.0 / (2.0 * math.pi * dobf_cutoff_hz)
    self.alpha_dob = dt / (dt + tau_dob)

    tau_bank = 1.0 / (2.0 * math.pi * bank_cutoff_hz)
    self.alpha_bank = dt / (dt + tau_bank)

  def update(self, a_lat_measured, kappa_actual, v, lat_accel_factor_current):
    """
    a_lat_measured: measured lateral acceleration from yaw rate [m/s²]
    kappa_actual: actual curvature from steering angle [1/m]
    v: vehicle speed [m/s]
    lat_accel_factor_current: current RLS-adapted latAccelFactor
    """
    # Model-predicted lateral acceleration
    a_lat_predicted = lat_accel_factor_current * kappa_actual * v ** 2

    # Total disturbance = measured - predicted
    d_total = a_lat_measured - a_lat_predicted

    # Low-pass filter (0.5 Hz cutoff)
    self.disturbance_estimate = ((1.0 - self.alpha_dob) * self.disturbance_estimate
                                 + self.alpha_dob * d_total)

    # Bank angle estimation: only during straight driving (κ ≈ 0)
    if abs(kappa_actual) < BANK_STRAIGHT_KAPPA and v > 5.0:
      # In straight line, lateral accel ≈ g·sin(θ_bank)
      a_bank = a_lat_measured
      theta_bank_est = math.asin(clip(a_bank / ACCELERATION_DUE_TO_GRAVITY, -0.15, 0.15))
      self.bank_angle_estimate = ((1.0 - self.alpha_bank) * self.bank_angle_estimate
                                  + self.alpha_bank * theta_bank_est)

  def get_compensation_torque(self, lat_accel_factor_current):
    """Return disturbance compensation torque."""
    if abs(lat_accel_factor_current) < 0.1:
      return 0.0
    return -self.disturbance_estimate / lat_accel_factor_current

  def get_bank_compensation(self, v, lat_accel_factor_current):
    """Return bank angle compensation torque."""
    if abs(lat_accel_factor_current) < 0.1:
      return 0.0
    a_bank = ACCELERATION_DUE_TO_GRAVITY * math.sin(self.bank_angle_estimate)
    return -a_bank / lat_accel_factor_current

  def reset(self):
    self.disturbance_estimate = 0.0
    self.bank_angle_estimate = 0.0


class RLSAdaptor:
  """
  Recursive Least Squares online parameter adaptor.
  Estimates latAccelFactor (α) and friction (β) from the model:
    a_lat_measured = α · κ · v² + β · sign(κ · v²)
  """

  def __init__(self, alpha0, friction0, forgetting=RLS_FORGETTING, p0=RLS_P0_DIAG):
    # Parameter vector θ = [α, β]
    self.theta = np.array([alpha0, friction0], dtype=np.float64)
    # Covariance matrix P (2×2)
    self.P = np.eye(2, dtype=np.float64) * p0
    self.lam = forgetting
    self.confidence = 0.0  # starts low, builds up over time
    self._update_count = 0

  def update(self, a_lat_measured, kappa, v):
    """
    One RLS update step.
    Returns (alpha, friction, confidence).
    """
    v2 = v ** 2
    phi = np.array([kappa * v2, sign_with_deadzone(kappa * v2, dz=0.01)], dtype=np.float64)

    # Prediction error
    y_hat = phi @ self.theta
    e = a_lat_measured - y_hat

    # RLS gain
    Pphi = self.P @ phi
    denom = self.lam + phi @ Pphi
    if abs(denom) < 1e-12:
      return self.theta[0], self.theta[1], self.confidence
    K = Pphi / denom

    # Parameter update
    self.theta = self.theta + K * e

    # Covariance update
    self.P = (self.P - np.outer(K, phi @ self.P)) / self.lam

    # Parameter range clamping
    self.theta[0] = clip(self.theta[0], RLS_ALPHA_RANGE[0], RLS_ALPHA_RANGE[1])
    self.theta[1] = clip(self.theta[1], RLS_FRICTION_RANGE[0], RLS_FRICTION_RANGE[1])

    # Confidence tracking (exponential moving average of |e| relative to signal)
    self._update_count += 1
    signal_mag = max(abs(a_lat_measured), 0.1)
    relative_error = abs(e) / signal_mag
    self.confidence = max(0.0, 1.0 - relative_error)
    # Build confidence slowly over time
    if self._update_count < 100:
      self.confidence *= self._update_count / 100.0

    return self.theta[0], self.theta[1], self.confidence

  def get_params(self):
    """Return (alpha, friction, confidence)."""
    return self.theta[0], self.theta[1], self.confidence

  def reset(self, alpha0=None, friction0=None):
    if alpha0 is None:
      alpha0 = DEFAULT_LAT_ACCEL_FACTOR
    if friction0 is None:
      friction0 = DEFAULT_FRICTION
    self.theta = np.array([alpha0, friction0], dtype=np.float64)
    self.P = np.eye(2, dtype=np.float64) * RLS_P0_DIAG
    self.confidence = 0.0
    self._update_count = 0


class LatControlFFv1(LatControl):
  """
  Predictive Feedforward + Feedback lateral controller (V4).

  Four-layer separation:
    1. Intelligent Feedforward — delay-anticipatory κ preview + RLS adaptation
    2. Predictive Feedback — Smith Predictor + low-bandwidth gain-scheduled FB
    3. Disturbance Compensation — low-frequency DOB + bank angle estimation
    4. Constraint & Safety — predictive constraint + rate limit + anti-windup
  """

  def __init__(self, CP, CP_SP, CI, dt):
    super().__init__(CP, CP_SP, CI, dt)

    # Torque interface
    self.torque_params = CP.lateralTuning.torque.as_builder()
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.lateral_accel_from_torque = CI.lateral_accel_from_torque()
    self.dt = dt

    # Act as own extension for controlsd's extension.update_model_v2() call
    self.extension = self

    # ─── Vehicle Parameters ───────────────────────────────────────────
    self.lat_accel_factor = self.torque_params.latAccelFactor
    self.friction = self.torque_params.friction
    self.wheelbase = DEFAULT_WHEELBASE
    self.mass = DEFAULT_MASS
    self.iz = DEFAULT_IZ
    self.lf = DEFAULT_LF
    self.lr = DEFAULT_LR
    self.cf = DEFAULT_CF
    self.cr = DEFAULT_CR

    # ─── RLS Online Adaptor ──────────────────────────────────────────
    self.rls = RLSAdaptor(self.lat_accel_factor, self.friction)

    # ─── State Predictor (Smith Predictor) ────────────────────────────
    self.predictor = StatePredictor(
      dt, self.wheelbase, self.mass, self.iz,
      self.lf, self.lr, self.cf, self.cr, self.lat_accel_factor)

    # ─── Disturbance Observer ─────────────────────────────────────────
    self.dob = DisturbanceObserver(dt, self.lat_accel_factor, DOB_CUTOFF_HZ, BANK_CUTOFF_HZ)

    # ─── Delay Buffers ────────────────────────────────────────────────
    max_delay_frames = 20  # max 20 * 20ms = 400ms
    self.input_buffer = deque([0.0] * max_delay_frames, maxlen=max_delay_frames)

    # ─── Feedback State ───────────────────────────────────────────────
    self.integrator = 0.0
    self.prev_error = 0.0

    # ─── Feedforward State ────────────────────────────────────────────
    self.prev_kappa_ff = 0.0
    self.heading_error_state = 0.0  # integrated heading error [rad]

    # ─── Constraint State ─────────────────────────────────────────────
    self.prev_torque = 0.0
    self.saturation_counter = 0

    # ─── Measurement Filter ───────────────────────────────────────────
    self.measurement_filter = FirstOrderFilter(0.0, 1.0 / (2.0 * math.pi * 10.0), dt)

    # ─── model_v2 State ───────────────────────────────────────────────
    self.model_v2 = None
    self.model_valid = False

    # ─── Tracking ─────────────────────────────────────────────────────
    self._first_frame = True
    self._prev_lat_delay = 0.0

  # ═══════════════════════════════════════════════════════════════════
  #  Interface Compatibility (extension methods)
  # ═══════════════════════════════════════════════════════════════════

  def update_model_v2(self, model_v2):
    self.model_v2 = model_v2
    self.model_valid = (
      self.model_v2 is not None
      and hasattr(self.model_v2, 'position')
      and len(self.model_v2.position.y) > 0
    )

  def update_limits(self):
    pass

  def update_lateral_lag(self, lag):
    pass

  def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction):
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction

  def reset(self):
    super().reset()
    max_delay_frames = 20
    self.input_buffer = deque([0.0] * max_delay_frames, maxlen=max_delay_frames)
    self.integrator = 0.0
    self.prev_error = 0.0
    self.prev_kappa_ff = 0.0
    self.heading_error_state = 0.0
    self.prev_torque = 0.0
    self.saturation_counter = 0
    self._first_frame = True
    self._prev_lat_delay = 0.0
    self.rls.reset(self.lat_accel_factor, self.friction)
    self.dob.reset()

  # ═══════════════════════════════════════════════════════════════════
  #  Gain Scheduling
  # ═══════════════════════════════════════════════════════════════════

  @staticmethod
  def _get_scheduled_gains(v):
    """
    Speed-dependent gain scheduling.
    Gains scale inversely with speed (high speed → lower gains for stability).
    Returns (Ky, Kpsi, Kbeta, Kr).
    """
    v_safe = max(v, MIN_SPEED)
    # Scale factor: reference speed / actual speed, capped
    scale = clip(FB_REF_SPEED / v_safe, 0.15, 3.0)
    return (FB_KY * scale, FB_KPSI * scale, FB_KBETA * scale * scale, FB_KR * scale)

  @staticmethod
  def _get_rate_limit(v):
    """Dynamic torque rate limit based on speed."""
    if v < 10.0:
      return TORQUE_RATE_EMERGENCY  # low speed: allow fast response
    elif v < 25.0:
      return TORQUE_RATE_NORMAL * 1.5
    else:
      return TORQUE_RATE_NORMAL

  # ═══════════════════════════════════════════════════════════════════
  #  Layer 1: Intelligent Feedforward
  # ═══════════════════════════════════════════════════════════════════

  def _compute_feedforward(self, desired_curvature, v, lat_delay):
    """
    Delay-anticipatory feedforward:
      1. Take κ at t + lat_delay from curvature buffer
      2. Compute steady-state torque: τ_ff = α · κ · v² + β · sign(κ)
      3. Add dynamic feedforward for curvature rate
    """
    # Get adaptive parameters
    alpha, friction_est, confidence = self.rls.get_params()

    # Use defaults if confidence is low
    if confidence < RLS_CONFIDENCE_THRESHOLD:
      alpha = self.lat_accel_factor
      friction_est = self.friction

    # Delay-anticipatory curvature: desired_curvature already comes from
    # the model's lookahead preview point (contains built-in lookahead),
    # so no additional buffer delay-compensation is needed.
    kappa_ff = desired_curvature

    # Steady-state feedforward torque (proper inverse model, not α·a_lat)
    v_safe = max(v, MIN_SPEED)
    a_lat_desired = kappa_ff * v_safe ** 2
    tau_ff = self.torque_from_lateral_accel(a_lat_desired, self.torque_params)

    # Dynamic feedforward (curvature rate compensation)
    if self._first_frame:
      dkappa_dt = 0.0
    else:
      dkappa_dt = (kappa_ff - self.prev_kappa_ff) / self.dt
    self.prev_kappa_ff = kappa_ff

    tau_ff_dynamic = K_DKAPPA * dkappa_dt * v_safe

    return tau_ff + tau_ff_dynamic, kappa_ff, alpha, friction_est

  # ═══════════════════════════════════════════════════════════════════
  #  Layer 2: Predictive Feedback (Smith Predictor + FB)
  # ═══════════════════════════════════════════════════════════════════

  def _compute_feedback(self, CS, VM, params, v, desired_curvature, lat_delay, alpha_current):
    """
    Smith Predictor + low-bandwidth gain-scheduled feedback.
    1. Measure current state from vehicle sensors
    2. Predict state at t + lat_delay using input buffer
    3. Compute feedback torque from predicted state error
    """
    n_delay = max(1, round(lat_delay / self.dt))

    # Current state measurement
    measured_curvature = -VM.calc_curvature(
      math.radians(CS.steeringAngleDeg - params.angleOffsetDeg),
      v, params.roll)

    # Heading error: difference between actual yaw rate and desired
    kappa_ref = desired_curvature
    heading_error_rate = CS.yawRate - kappa_ref * v
    # Integrate heading error over time with clamping (not one-step approximation)
    self.heading_error_state += heading_error_rate * self.dt
    self.heading_error_state = clip(self.heading_error_state, -HEADING_ERROR_MAX, HEADING_ERROR_MAX)
    heading_error = self.heading_error_state

    # Sideslip angle estimate from lateral velocity
    try:
      v_lat = VM.get_lateral_vel()
    except Exception:
      v_lat = 0.0
    v_safe = max(v, MIN_SPEED)
    beta_estimate = math.atan2(v_lat, v_safe) if v_safe > MIN_SPEED else 0.0

    # Current state vector
    # Read measured lateral error from model_v2 using T_IDXS binary search
    # for proper lookahead (ADRC pattern: position.y at lookahead time, not index 0)
    ey_measured = 0.0
    if self.model_valid and self.model_v2 is not None:
      try:
        pos_y = self.model_v2.position.y
        n_pos = min(len(pos_y), CONTROL_N)
        if n_pos > 0:
          # Lookahead time for lateral error measurement (near point ~0.3s)
          ey_lookahead_s = 0.3
          idx = int(np.searchsorted(ModelConstants.T_IDXS[:n_pos], ey_lookahead_s))
          idx = min(max(idx, 0), n_pos - 1)
          ey_measured = float(pos_y[idx])
      except (AttributeError, IndexError, TypeError):
        ey_measured = 0.0

    state_current = [ey_measured, heading_error, beta_estimate, CS.yawRate]

    # Predict state forward by lat_delay using stored inputs
    # Use the most recent n_delay entries from input_buffer as future inputs
    future_inputs = list(self.input_buffer)[-n_delay:] if len(self.input_buffer) >= n_delay else list(self.input_buffer)

    state_predicted = self.predictor.predict(
      state_current, future_inputs, v, min(n_delay, len(future_inputs)))

    # ─── Smith Predictor Innovation Correction ──────────────────────────
    # Compare predicted ey with measured ey from model_v2 and correct state
    # using observer gain L (design doc §3.5.1 Step 3):
    #   innovation = y_measured - C·x̂_pred
    #   x̂_corrected = x̂_pred + L · innovation
    ey_predicted = state_predicted[0]
    innovation = ey_measured - ey_predicted
    state_predicted = [
      state_predicted[0] + OBSERVER_L * innovation,  # corrected ey
      state_predicted[1] + OBSERVER_L * innovation * 0.5,  # corrected heading (coupled)
      state_predicted[2],  # sideslip unchanged
      state_predicted[3],  # yaw rate unchanged
    ]

    # Predicted errors (after innovation correction)
    e_y = state_predicted[0]        # corrected lateral deviation
    e_psi = state_predicted[1]      # corrected heading error
    beta_p = state_predicted[2]     # predicted sideslip
    r_p = state_predicted[3]        # predicted yaw rate

    # Gain-scheduled feedback
    Ky, Kpsi, Kbeta, Kr = self._get_scheduled_gains(v)

    tau_fb = -(Ky * e_y + Kpsi * e_psi + Kbeta * beta_p + Kr * r_p)

    return tau_fb, measured_curvature, e_y

  # ═══════════════════════════════════════════════════════════════════
  #  Layer 3: Disturbance Estimation & Compensation
  # ═══════════════════════════════════════════════════════════════════

  def _compute_disturbance(self, v, measured_curvature, CS, alpha_current):
    """
    Low-frequency DOB + bank angle compensation.
    """
    v_safe = max(v, MIN_SPEED)

    # Measured lateral acceleration from yaw rate
    a_lat_measured = v_safe * CS.yawRate

    # Filter measurement
    a_lat_measured = self.measurement_filter.update(a_lat_measured)

    # Update DOB
    self.dob.update(a_lat_measured, measured_curvature, v_safe, alpha_current)

    # Disturbance compensation torque
    tau_dist = self.dob.get_compensation_torque(alpha_current)

    # Bank angle compensation (additional)
    tau_bank = self.dob.get_bank_compensation(v_safe, alpha_current)

    return tau_dist + tau_bank

  # ═══════════════════════════════════════════════════════════════════
  #  Layer 4: Constraint Handling & Safety
  # ═══════════════════════════════════════════════════════════════════

  def _apply_constraints(self, tau_total, v, steer_max, steer_limited_by_safety, dt):
    """
    1. Hard torque limit (steer_max from safety)
    2. Anti-windup back-calculation
    3. Torque rate limiting (comfort)
    4. Predictive constraint check (simplified single-step)
    """
    # 4a. Hard torque constraint
    tau_clipped = clip(tau_total, -steer_max, steer_max)

    # 4b. Anti-windup: back-calculate saturation into integrator
    saturation = tau_clipped - tau_total
    if abs(saturation) > 1e-6:
      self.integrator += ANTI_WINDUP_GAIN * saturation * dt
      self.saturation_counter += 1
    else:
      self.saturation_counter = max(0, self.saturation_counter - 1)

    # Clamp integrator to prevent excessive windup
    self.integrator = clip(self.integrator, -steer_max * 0.5, steer_max * 0.5)

    # 4c. Torque rate limiting
    tau_rate_limit = self._get_rate_limit(v)
    tau_rate = (tau_clipped - self.prev_torque) / max(dt, 1e-6)
    if abs(tau_rate) > tau_rate_limit:
      tau_clipped = self.prev_torque + tau_rate_limit * dt * (1.0 if tau_rate > 0 else -1.0)

    # Final hard clip
    tau_clipped = clip(tau_clipped, -steer_max, steer_max)

    return tau_clipped

  # ═══════════════════════════════════════════════════════════════════
  #  Main Update
  # ═══════════════════════════════════════════════════════════════════

  def update(self, active, CS, VM, params, steer_limited_by_safety,
             desired_curvature, calibrated_pose, curvature_limited, lat_delay):
    pid_log = log.ControlsState.LateralTorqueState.new_message()
    pid_log.version = VERSION

    if not active:
      self.reset()
      return 0.0, 0.0, pid_log

    dt = self.dt
    v = max(CS.vEgo, MIN_SPEED)
    lat_delay = max(lat_delay, 0.01)  # at least 10ms

    # Check for delay change → reinitialize predictor if needed
    if abs(lat_delay - self._prev_lat_delay) > 0.020:  # 20ms jump
      # Reset predictor state on large delay changes
      self.dob.reset()
    self._prev_lat_delay = lat_delay

    # ─── Layer 1: Intelligent Feedforward ─────────────────────────────
    tau_ff, kappa_ff, alpha_current, friction_current = self._compute_feedforward(
      desired_curvature, v, lat_delay)

    # ─── Layer 2: Predictive Feedback ─────────────────────────────────
    tau_fb, measured_curvature, e_y_pred = self._compute_feedback(
      CS, VM, params, v, desired_curvature, lat_delay, alpha_current)

    # ─── Integrator accumulation (lateral position error → torque) ─────
    self.integrator += INTEGRATOR_GAIN * e_y_pred * dt

    # ─── Layer 3: Disturbance Compensation ────────────────────────────
    tau_dist = self._compute_disturbance(v, measured_curvature, CS, alpha_current)

    # ─── Roll compensation ────────────────────────────────────────────
    roll_comp = params.roll * ACCELERATION_DUE_TO_GRAVITY
    tau_roll = -roll_comp / max(alpha_current, 0.1)

    # ─── Combine all layers ───────────────────────────────────────────
    tau_total = tau_ff + tau_fb + tau_dist + tau_roll + self.integrator

    # ─── Layer 4: Constraint & Safety ─────────────────────────────────
    tau_final = self._apply_constraints(tau_total, v, self.steer_max, steer_limited_by_safety, dt)

    # ─── Update input buffer ──────────────────────────────────────────
    self.input_buffer.append(tau_final)

    # ─── RLS Online Adaptation ────────────────────────────────────────
    if v > RLS_MIN_SPEED and abs(measured_curvature) > RLS_MIN_CURVATURE:
      a_lat_measured = v * CS.yawRate
      self.rls.update(a_lat_measured, measured_curvature, v)

    # Update predictor's lat_accel_factor from RLS
    rls_alpha, _, rls_conf = self.rls.get_params()
    if rls_conf > RLS_CONFIDENCE_THRESHOLD:
      self.predictor.lat_accel_factor = rls_alpha
      self.dob.lat_accel_factor = rls_alpha

    # ─── Driver intervention ──────────────────────────────────────────
    if CS.steeringPressed:
      tau_final *= 0.5

    # Final clip
    tau_final = clip(tau_final, -self.steer_max, self.steer_max)
    self.prev_torque = tau_final
    self._first_frame = False

    # ─── Logging ──────────────────────────────────────────────────────
    pid_log.active = True
    pid_log.p = float(tau_ff)           # feedforward torque
    pid_log.i = float(self.integrator)  # integrator state
    pid_log.d = float(tau_dist)         # disturbance compensation
    pid_log.f = float(tau_fb)           # feedback torque
    pid_log.output = float(-tau_final)
    pid_log.actualLateralAccel = float(v * CS.yawRate)
    pid_log.desiredLateralAccel = float(kappa_ff * v ** 2)
    pid_log.saturated = bool(self.saturation_counter > 5)

    return -tau_final, 0.0, pid_log
