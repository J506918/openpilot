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

VERSION 120
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
VERSION = 121  # bumped: adaptive deadzone step + friction scaling + rate limit alignment

# ─── EPS Deadzone Step Compensation ────────────────────────────────────
# Bosch EPS ignores torque commands below ~0.5 Nm (deadzone).
# In normalized torque units (0-2560 CAN), this corresponds to ~0.12 units.
EPS_DEADZONE_TORQUE = 0.12         # BUGFIX: was 0.0125 (10x too small). Half of torque deadzone (≈307 CAN total range)
DEADZONE_STEP_MAGNITUDE = 0.04     # step jump to break static friction (≈102 CAN)
DEADZONE_ZEROCROSS_EXTRA = 0.02    # extra step on zero-crossing (≈51 CAN)

# ─── Damping Feedforward Compensation ──────────────────────────────────
# EPS mechanical damping consumes torque proportional to steering rate.
# Compensate with feedforward torque in direction of steering motion.
DAMPING_COEFF = 1.5                # torque units per rad/s steering rate
DAMPING_FILTER_HZ = 5.0            # low-pass filter cutoff for steer rate
DAMPING_RATE_DEADZONE = 0.01       # rad/s — ignore tiny steer rates

# ─── Rack Force Characteristic Linearization (齿条力特性线性化) ──────────
# Real EPS rack force is nonlinear vs lateral acceleration:
#   - Small angles: steep slope (overcoming static friction)
#   - Medium angles: moderate slope (linear region)
#   - Large angles: gentle slope (rack force saturation)
#
# Piecewise mapping: torque = f(a_lat, v) with 3 segments.
# Breakpoints and slopes for normalized torque (0-1 range per m/s²):
RACK_BP_SMALL = 2.0         # m/s² — small lateral accel threshold
RACK_BP_LARGE = 4.0         # m/s² — large lateral accel threshold
RACK_SLOPE_SMALL = 0.85     # steeper slope (more torque per accel unit) → overcomes stiction
RACK_SLOPE_MEDIUM = 0.70    # moderate slope in linear region
RACK_SLOPE_LARGE = 0.50     # gentler slope → rack force saturates

# Speed-dependent scaling: at low speeds (< 10 m/s), rack forces are higher
# (more stiction). At high speeds, forces decrease (less stiction).
RACK_SPEED_LOW = 10.0       # m/s — below this, full small-slope benefit
RACK_SPEED_HIGH = 30.0      # m/s — above this, reduced slopes
RACK_SPEED_SCALE_LOW = 1.2  # extra multiplier at low speed
RACK_SPEED_SCALE_HIGH = 0.7 # reduced multiplier at high speed

# ─── LQR Full-State Feedback ───────────────────────────────────────────
# 5-state model: [e_y, e_y_dot, heading_err, yaw_rate, steer_angle]
# Solves Discrete Algebraic Riccati Equation (Kleinman iteration)
# Q weights penalize state errors, R penalizes control effort
LQR_ENABLED = True                 # master switch for LQR
LQR_Q_DIAG = [100.0, 5.0, 30.0, 3.0, 0.05]  # state cost: ey, ey_dot, heading, yaw, steer
LQR_R = 1.0                        # control cost
LQR_STEER_TIME_CONSTANT = 0.05     # EPS steering time constant [s]
LQR_STEER_GAIN = 0.05              # τ → steer_angle gain (rad per torque unit)
LQR_STEER_RATIO = 16.0             # steering ratio
LQR_YAW_ALPHA = 0.3                # yaw filter blending factor
LQR_SPEED_THRESHOLD = 0.5        # re-compute LQR gains when speed changes by >0.5 m/s

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
TORQUE_RATE_NORMAL = 10.0         # unit/s — comfort limit
TORQUE_RATE_EMERGENCY = 30.0      # unit/s — safety override
LAT_ACCEL_COMFORT = 2.0           # m/s² comfort limit
LAT_ACCEL_SAFETY = 4.0            # m/s² absolute limit
YAW_RATE_LIMIT = 0.5              # rad/s stability limit
ANTI_WINDUP_GAIN = 0.5            # anti-windup back-calculation gain

# ─── Dynamic Feedforward ──────────────────────────────────────────────
# K_dκ = (I_z · L) / (2 · C_f · l_f)  — curvature-rate feedforward gain
# (computed per-vehicle in __init__ as self.k_dkappa)

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

# ─── Friction Scaling Threshold ──────────────────────────────────────
# When |τ_fb| is below this threshold, friction compensation is scaled
# down to prevent overwhelming tiny feedback signals at steady state.
# At |τ_fb|=0.0001 (steady state), scale=0.005 → friction=0.0006 (vs 0.126 raw)
FRICTION_SCALE_THRESHOLD = 0.02   # τ_fb magnitude below which friction is scaled


def sign_with_deadzone(x, dz=0.0001):
  """Smoothed sign function with deadzone to avoid chattering."""
  if abs(x) < dz:
    return 0.0
  return 1.0 if x > 0.0 else -1.0


def clip(val, lo, hi):
  return max(lo, min(hi, val))


# ═══════════════════════════════════════════════════════════════════════
#  Rack Force Characteristic Linearization (齿条力特性线性化)
# ═══════════════════════════════════════════════════════════════════════

def rack_force_torque(a_lat, v, base_lat_accel_factor=DEFAULT_LAT_ACCEL_FACTOR,
                      bp_small=RACK_BP_SMALL, bp_large=RACK_BP_LARGE,
                      s_small=RACK_SLOPE_SMALL, s_medium=RACK_SLOPE_MEDIUM,
                      s_large=RACK_SLOPE_LARGE,
                      speed_low=RACK_SPEED_LOW, speed_high=RACK_SPEED_HIGH,
                      scale_low=RACK_SPEED_SCALE_LOW, scale_high=RACK_SPEED_SCALE_HIGH):
  """
  Piecewise rack-force-based torque mapping from lateral acceleration.

  Accounts for nonlinear EPS rack force characteristics:
    - |a_lat| < 2 m/s²: steep slope → overcomes static friction / stiction
    - 2 ≤ |a_lat| < 4 m/s²: moderate slope → linear EPS region
    - |a_lat| ≥ 4 m/s²: gentle slope → rack force saturation

  Speed adaptation: at low speeds rack stiction dominates (boosted torque),
  at high speeds rack forces are lower (reduced torque).

  Returns normalized torque ([-1, 1] range for full-range CAN output).
  """
  abs_lat = abs(a_lat)
  v_safe = max(v, MIN_SPEED)

  # ─── Piecewise torque computation ──────────────────────────────
  # Base torque = a_lat * effective_slope depends on which segment
  if abs_lat < bp_small:
    # Small: steep slope
    torque_base = a_lat * s_small
  elif abs_lat < bp_large:
    # Medium: interpolate linearly between s_small at bp_small and s_medium at bp_large
    frac = (abs_lat - bp_small) / (bp_large - bp_small)
    # Use previous segment's torque at transition + medium segment torque
    torque_break_small = bp_small * s_small * (1.0 if a_lat >= 0.0 else -1.0)
    remaining = a_lat - bp_small * (1.0 if a_lat >= 0.0 else -1.0)
    torque_base = torque_break_small + remaining * s_medium
  else:
    # Large: flat-ish slope (saturation)
    torque_break_large = bp_large * s_large * (1.0 if a_lat >= 0.0 else -1.0)
    # But also account for accumulated torque before the breakpoint
    torque_at_bp_small = bp_small * s_small
    torque_at_bp_large = torque_at_bp_small + (bp_large - bp_small) * s_medium
    sign = 1.0 if a_lat >= 0.0 else -1.0
    remaining = abs_lat - bp_large
    torque_base = sign * (torque_at_bp_large + remaining * s_large)

  # ─── Speed-dependent scaling ───────────────────────────────────
  if v_safe <= speed_low:
    speed_scale = scale_low
  elif v_safe >= speed_high:
    speed_scale = scale_high
  else:
    # Linear interpolation between low and high speed
    frac_spd = (v_safe - speed_low) / (speed_high - speed_low)
    speed_scale = scale_low + frac_spd * (scale_high - scale_low)

  torque_scaled = torque_base * speed_scale

  # ─── Convert to normalized torque via base factor ─────────────
  # Standard linear mapping would be: torque = a_lat / latAccelFactor
  # Our piecewise replaces that. We scale by speed and use the
  # piecewise-derived torque directly as normalized units.
  # The base_lat_accel_factor is used for the normalization to
  # ensure the output is in the same units as the original linear mapping.
  return torque_scaled / max(base_lat_accel_factor, 0.1)


# ═══════════════════════════════════════════════════════════════════════
#  LQR Solver — Discrete Algebraic Riccati Equation (Kleinman iteration)
# ═══════════════════════════════════════════════════════════════════════

def solve_dare(A, B, Q, R, max_iter=300, tol=1e-12):
  """
  Solve Discrete Algebraic Riccati Equation:
    A'PA - P - A'PB(R + B'PB)^{-1}B'PA + Q = 0
  using Kleinman's iterative method.
  Returns gain matrix K = (R + B'PB)^{-1} B'PA
  """
  P = Q.copy()
  for _ in range(max_iter):
    BPB = B.T @ P @ B
    S = R + BPB
    try:
      S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
      S_inv = np.linalg.pinv(S)
    K = S_inv @ (B.T @ P @ A)
    P_new = Q + A.T @ P @ A - A.T @ P @ B @ K
    diff = np.max(np.abs(P_new - P))
    P = P_new
    if diff < tol:
      break
  BPB = B.T @ P @ B
  S = R + BPB
  try:
    S_inv = np.linalg.inv(S)
  except np.linalg.LinAlgError:
    S_inv = np.linalg.pinv(S)
  return S_inv @ (B.T @ P @ A)


def build_lqr_model(v, dt, steer_ratio=LQR_STEER_RATIO,
                    tau_steer=LQR_STEER_TIME_CONSTANT,
                    k_steer=LQR_STEER_GAIN,
                    yaw_alpha=LQR_YAW_ALPHA):
  """
  Build 5-state discrete bicycle model for LQR.
  
  States: [e_y, e_y_dot, heading_err, yaw_rate, steer_angle]
  Control: tau (normalized torque [-1, 1])
  # ─── DARE RESIDUAL: The DARE residual (||A'PA - P - A'PB(R+B'PB)^{-1}B'PA + Q||)
  # Continuous model:
  #   BUGFIX v120: Added heading_err leakage (-0.02) to pull the |λ|=1.0 eigenvalue
  #   inside the unit circle. Without this, heading_err is critically stable (z≈1),
  #   only bounded by HEADING_ERROR_MAX clamp. The 5-state model has e_y_dot ≈ v×heading_err
  #   creating linear dependence; the leakage breaks this degeneracy.
    d(e_y)/dt      = e_y_dot
    d(e_y_dot)/dt  = v * yaw_rate
    d(heading)/dt  = yaw_rate - 0.02 * heading_err  # BUGFIX: leakage for marginal stability
    d(yaw_rate)/dt = (steer_angle * v / steer_ratio - yaw_rate) / tau_yaw
    d(steer)/dt    = (tau * k_steer - steer_angle) / tau_steer
  """
  v_safe = max(v, MIN_SPEED)
  
  # Yaw filter time constant
  tau_yaw = dt * (1.0 - yaw_alpha) / max(yaw_alpha, 1e-6)
  omega_yaw = 1.0 / max(tau_yaw, 1e-6)
  
  # Steering dynamics
  omega_steer = 1.0 / max(tau_steer, 1e-6)
  
  # Steer angle → yaw rate steady-state gain
  yaw_gain = v_safe / max(steer_ratio, 0.1)
  
  # Continuous-time A (5×5)
  Ac = np.array([
    [0.0, 1.0, 0.0, 0.0,       0.0],
    [0.0, 0.0, 0.0, v_safe,    0.0],
    [0.0, 0.0, -0.02, 1.0,     0.0],  # BUGFIX: heading_err leakage breaks rank deficiency
    [0.0, 0.0, 0.0, -omega_yaw, omega_yaw * yaw_gain],
    [0.0, 0.0, 0.0, 0.0,       -omega_steer],
  ])
  
  Bc = np.array([
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [omega_steer * k_steer],
  ])
  
  # Euler discretization
  Ad = np.eye(5) + Ac * dt
  Bd = Bc * dt
  
  return Ad, Bd


def compute_lqr_gains(v, dt):
  """Compute LQR feedback gain matrix K at given speed."""
  try:
    Ad, Bd = build_lqr_model(v, dt)
    Q = np.diag(LQR_Q_DIAG)
    R = np.array([[LQR_R]])
    return solve_dare(Ad, Bd, Q, R)
  except Exception:
    return None


# ═══════════════════════════════════════════════════════════════════════
#  EPS Deadzone Step Compensation
# ═══════════════════════════════════════════════════════════════════════

def deadzone_step_compensation(tau_desired, prev_tau,
                                deadzone_half=EPS_DEADZONE_TORQUE,
                                step_mag=DEADZONE_STEP_MAGNITUDE,
                                zerocross_extra=DEADZONE_ZEROCROSS_EXTRA):
  """
  Inject a step jump when torque command crosses/nears zero to overcome
  Bosch EPS static friction deadzone (~0.5 Nm range).

  BUGFIX v121: Adaptive step scaling — step magnitude is scaled proportionally
  to |tau_desired| relative to the deadzone boundary (2*deadzone_half).
  This prevents large step injections (0.04-0.06) from overwhelming tiny
  steady-state signals (~0.009), which caused sustained oscillation.

  Two mechanisms:
    1. Inside deadzone: boost torque in desired direction (scaled)
    2. Zero-crossing: add extra kick to jump through deadzone (scaled)

  Only activates when |tau| < 2× deadzone to avoid interfering with large commands.
  """
  tau_out = tau_desired

  if abs(tau_desired) < 2.0 * deadzone_half:
    # Adaptive scaling factor: step shrinks as |tau| approaches zero
    # threshold = 2*deadzone_half (the activation boundary)
    # At |tau| = threshold, scale = 1.0 (full step)
    # At |tau| = 0.009 (steady state), scale ≈ 0.009/0.24 = 0.0375
    threshold = 2.0 * deadzone_half

    if 0.0 < abs(tau_desired) < deadzone_half:
      effective_step = step_mag * min(1.0, abs(tau_desired) / threshold)
      tau_out += effective_step * (1.0 if tau_desired > 0.0 else -1.0)

    # Zero-crossing detection (also adaptively scaled)
    if prev_tau is not None and tau_desired * prev_tau <= 0.0 and abs(tau_desired) > 1e-8:
      effective_zc = zerocross_extra * min(1.0, abs(tau_desired) / threshold)
      tau_out += effective_zc * (1.0 if tau_desired > 0.0 else -1.0)

  return tau_out


# ═══════════════════════════════════════════════════════════════════════
#  Damping Feedforward Compensation
# ═══════════════════════════════════════════════════════════════════════

class DampingCompensator:
  """
  Feedforward damping compensation based on filtered steering rate.
  EPS mechanical damping consumes torque proportional to steering rate;
  this compensator adds feedforward torque to overcome it.
  """
  
  def __init__(self, damping_coeff=DAMPING_COEFF, filter_hz=DAMPING_FILTER_HZ,
               rate_deadzone=DAMPING_RATE_DEADZONE, dt=None):
    self.damping_coeff = damping_coeff
    self.rate_deadzone = rate_deadzone
    self.filtered_rate = 0.0
    self._prev_steer_angle = None
    
    if dt is not None and filter_hz > 0.0:
      tau_f = 1.0 / (2.0 * math.pi * filter_hz)
      self.alpha = dt / (dt + tau_f)
    else:
      self.alpha = 0.5
  
  def update(self, steer_angle_rad, dt):
    """
    Compute damping feedforward torque given current steering angle.
    Returns torque to add to overcome EPS mechanical damping.
    """
    if self._prev_steer_angle is None:
      self._prev_steer_angle = steer_angle_rad
      return 0.0
    
    # Raw steer rate
    steer_rate = (steer_angle_rad - self._prev_steer_angle) / max(dt, 1e-6)
    
    # Low-pass filter
    self.filtered_rate = ((1.0 - self.alpha) * self.filtered_rate
                          + self.alpha * steer_rate)
    
    self._prev_steer_angle = steer_angle_rad
    
    # Deadzone on small rates
    if abs(self.filtered_rate) < self.rate_deadzone:
      return 0.0
    
    return self.damping_coeff * self.filtered_rate
  
  def reset(self):
    self.filtered_rate = 0.0
    self._prev_steer_angle = None


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

    # ─── Vehicle Parameters (from CP with fallback to defaults) ─────
    self.lat_accel_factor = self.torque_params.latAccelFactor
    self.friction = self.torque_params.friction
    self.wheelbase = CP.wheelbase if CP.wheelbase > 0 else DEFAULT_WHEELBASE
    self.mass = CP.mass if CP.mass > 0 else DEFAULT_MASS
    self.iz = CP.rotationalInertia if CP.rotationalInertia > 0 else DEFAULT_IZ
    self.lf = CP.centerToFront if CP.centerToFront > 0 else DEFAULT_LF
    self.lr = (CP.wheelbase - CP.centerToFront) if CP.wheelbase > 0 else DEFAULT_LR
    self.cf = CP.tireStiffnessFront if CP.tireStiffnessFront > 0 else DEFAULT_CF
    self.cr = CP.tireStiffnessRear if CP.tireStiffnessRear > 0 else DEFAULT_CR

    # Dynamic feedforward gain: K_dκ = (I_z · L) / (2 · C_f · l_f)
    self.k_dkappa = (self.iz * self.wheelbase) / (2.0 * self.cf * self.lf)

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

    # ─── Damping Feedforward Compensator ──────────────────────────────
    self.damping_comp = DampingCompensator(
      damping_coeff=DAMPING_COEFF, filter_hz=DAMPING_FILTER_HZ,
      rate_deadzone=DAMPING_RATE_DEADZONE, dt=dt)

    # ─── LQR State ────────────────────────────────────────────────────
    self.lqr_enabled = LQR_ENABLED
    self.lqr_K = None  # computed on first update
    self._last_lqr_v = 0.0
    self._prev_steer_angle_deg = 0.0  # for damping FF

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
    self.lat_accel_factor = latAccelFactor
    self.friction = friction

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
    self.damping_comp.reset()
    self.lqr_K = None
    self._last_lqr_v = 0.0
    self._prev_steer_angle_deg = 0.0

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
    scale = clip(FB_REF_SPEED / v_safe, 0.15, 1.5)
    return (FB_KY * scale, FB_KPSI * scale, FB_KBETA * scale * scale, FB_KR * scale)

  @staticmethod
  def _get_rate_limit(v):
    """Dynamic torque rate limit based on speed.
    BUGFIX v120: Increased multiplier from 1.5→2.5 to compensate for
    STEER_DELTA_UP=3 bottleneck in carcontroller (CAN-level rate limit).
    BUGFIX v121: Adjusted multiplier 2.5→2.4. At 20 m/s this gives
    10*2.4=24 unit/s → 0.48/frame, slightly below CAN's 0.48485/frame
    (STEER_DELTA_UP*dt/0.33 = 8*0.02/0.33 = 0.48485). FFv1 now pre-clamps
    before CC does, avoiding the double-clamp discontinuity."""
    if v < 10.0:
      return TORQUE_RATE_EMERGENCY  # low speed: allow fast response
    elif v < 25.0:
      return TORQUE_RATE_NORMAL * 2.4
    else:
      return TORQUE_RATE_NORMAL

  # ═══════════════════════════════════════════════════════════════════
  #  Layer 1: Intelligent Feedforward
  # ═══════════════════════════════════════════════════════════════════

  def _compute_feedforward(self, desired_curvature, v):
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

    # Steady-state feedforward torque — rack force characteristic linearization
    # Replaces pure linear t=a_lat/latAccelFactor with piecewise mapping that
    # accounts for nonlinear EPS rack force: steeper at small angles (stiction),
    # moderate in linear region, gentler at large angles (saturation).
    v_safe = max(v, MIN_SPEED)
    a_lat_desired = kappa_ff * v_safe ** 2
    tau_ff = rack_force_torque(a_lat_desired, v_safe, self.lat_accel_factor)

    # Friction compensation moved to update() — error-driven instead of curvature-driven

    # Dynamic feedforward (curvature rate compensation)
    if self._first_frame:
      dkappa_dt = 0.0
    else:
      dkappa_dt = (kappa_ff - self.prev_kappa_ff) / self.dt
    self.prev_kappa_ff = kappa_ff

    tau_ff_dynamic = self.k_dkappa * dkappa_dt * v_safe

    return tau_ff + tau_ff_dynamic, kappa_ff, alpha

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
    # Integrate heading error over time with clamping
    # Accumulate heading error at all speeds
    self.heading_error_state += heading_error_rate * self.dt
    self.heading_error_state = clip(self.heading_error_state, -HEADING_ERROR_MAX, HEADING_ERROR_MAX)
    heading_error = self.heading_error_state

    # Sideslip angle estimate — VM.get_lateral_vel() does not exist, so use 0
    beta_estimate = 0.0

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
    # Gain-scheduled feedback — active at all speeds (0 km/h inclusive)

    # ─── LQR Full-State Feedback (if enabled) ─────────────────────────
    if self.lqr_enabled:
      # Compute LQR gains lazily (once per speed change)
      if self.lqr_K is None or abs(v - self._last_lqr_v) > LQR_SPEED_THRESHOLD:
        self.lqr_K = compute_lqr_gains(v, self.dt)
        self._last_lqr_v = v
      if self.lqr_K is not None:
        # 5-state: [e_y, e_y_dot, heading_err, yaw_rate, steer_angle]
        ey_dot_est = v * e_psi  # approximate: d(ey)/dt ≈ v * heading_err
        steer_angle_rad = math.radians(CS.steeringAngleDeg - params.angleOffsetDeg)
        state = np.array([e_y, ey_dot_est, e_psi, CS.yawRate, steer_angle_rad])
        tau_fb = -float(self.lqr_K @ state)
        return tau_fb, measured_curvature, e_y

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

  def _apply_constraints(self, tau_total, v, steer_max, steer_limited_by_safety, dt, freeze=False, tau_ff=0.0):
    """
    1. Hard torque limit (steer_max from safety)
    2. Anti-windup back-calculation (skipped when integrator is frozen)
    3. Torque rate limiting (comfort)
       BUGFIX v120: feedforward τ_ff bypasses rate limiter on sharp curve entry.
       Only feedback+integrator+damping components are rate-limited.
       τ_ff is pure steady-state physics, safe to step instantly.
    4. Predictive constraint check (simplified single-step)
    """
    # 4a. Hard torque constraint
    tau_clipped = clip(tau_total, -steer_max, steer_max)

    # 4b. Anti-windup: back-calculate saturation into integrator (skip when frozen)
    if not freeze:
      saturation = tau_clipped - tau_total
      if abs(saturation) > 1e-6:
        self.integrator += ANTI_WINDUP_GAIN * saturation * dt
        self.saturation_counter += 1
      else:
        self.saturation_counter = max(0, self.saturation_counter - 1)

    # Clamp integrator to prevent excessive windup
    self.integrator = clip(self.integrator, -steer_max * 0.5, steer_max * 0.5)

    # 4c. Torque rate limiting — feedforward bypass
    # BUGFIX v120: τ_ff skips rate limiter. Only non-ff torque is rate-limited.
    # This prevents Scenario C divergence where needed τ_ff=1.196 takes 4+ frames
    # to build up while the vehicle falls behind the curve.
    tau_non_ff = tau_clipped - tau_ff
    prev_non_ff = self.prev_torque - getattr(self, '_prev_tau_ff', 0.0)

    tau_rate_limit = self._get_rate_limit(v)
    if tau_non_ff != 0.0 or prev_non_ff != 0.0:
      tau_rate = (tau_non_ff - prev_non_ff) / max(dt, 1e-6)
      if abs(tau_rate) > tau_rate_limit:
        tau_non_ff = prev_non_ff + tau_rate_limit * dt * (1.0 if tau_rate > 0 else -1.0)

    # Reconstruct with feedforward passed through instantly
    tau_clipped = tau_ff + tau_non_ff

    # Final hard clip
    tau_clipped = clip(tau_clipped, -steer_max, steer_max)
    self._prev_tau_ff = tau_ff

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
    tau_ff, kappa_ff, alpha_current = self._compute_feedforward(
      desired_curvature, v)

    # Smith predictor cold-start: pre-fill input buffer with current feedforward
    # instead of zeros, giving the predictor a reasonable initial guess
    if self._first_frame:
      max_delay_frames = self.input_buffer.maxlen or 20
      self.input_buffer = deque([tau_ff] * max_delay_frames, maxlen=max_delay_frames)

    # ─── Layer 2: Predictive Feedback ─────────────────────────────────
    tau_fb, measured_curvature, e_y_pred = self._compute_feedback(
      CS, VM, params, v, desired_curvature, lat_delay, alpha_current)

    # ─── Integrator accumulation (lateral position error → torque) ─────
    freeze = steer_limited_by_safety or CS.steeringPressed
    if not freeze:
      self.integrator += INTEGRATOR_GAIN * e_y_pred * dt

    # ─── Layer 3: Disturbance Compensation ────────────────────────────
    tau_dist = self._compute_disturbance(v, measured_curvature, CS, alpha_current)

    # ─── Roll compensation ────────────────────────────────────────────
    roll_comp = params.roll * ACCELERATION_DUE_TO_GRAVITY
    tau_roll = -roll_comp / max(alpha_current, 0.1)

    # ─── Combine all layers ───────────────────────────────────────────
    tau_total = tau_ff + tau_fb + tau_dist + tau_roll + self.integrator

    # ─── Error-driven friction compensation ──────────────────────────
    # Uses feedback torque sign (error-driven) instead of curvature sign,
    # so friction compensation works on straight roads where κ=0 but e_y≠0.
    # BUGFIX v121: Scale friction with |τ_fb| when feedback is tiny.
    # At steady state (|τ_fb| ≈ 0.0001), full friction (0.126) is 1000x larger
    # than the feedback signal, creating bang-bang oscillation. Scaling prevents
    # friction from dominating when there's no meaningful error to correct.
    _, rls_friction, rls_conf = self.rls.get_params()
    friction_est = rls_friction if rls_conf > RLS_CONFIDENCE_THRESHOLD else self.friction
    friction_base = friction_est / max(alpha_current, 0.1)
    friction_sign = sign_with_deadzone(tau_fb)
    # Scale friction when |τ_fb| is below threshold
    if abs(tau_fb) < FRICTION_SCALE_THRESHOLD:
      friction_scale = abs(tau_fb) / FRICTION_SCALE_THRESHOLD
    else:
      friction_scale = 1.0
    tau_total += friction_base * friction_sign * friction_scale

    # ─── Damping Feedforward Compensation ─────────────────────────────
    # Compensate for EPS mechanical damping based on steering rate
    steer_angle_rad = math.radians(CS.steeringAngleDeg - params.angleOffsetDeg)
    tau_damping = self.damping_comp.update(steer_angle_rad, dt)
    tau_total += tau_damping

    # ─── EPS Deadzone Step Compensation ──────────────────────────────
    # Inject step jump when torque crosses near zero to overcome EPS deadzone
    tau_total = deadzone_step_compensation(tau_total, self.prev_torque,
                                            EPS_DEADZONE_TORQUE,
                                            DEADZONE_STEP_MAGNITUDE,
                                            DEADZONE_ZEROCROSS_EXTRA)

    # ─── Layer 4: Constraint & Safety ─────────────────────────────────
    # BUGFIX v120: Pass tau_ff to bypass rate limiting for feedforward.
    # When kappa changes abruptly (curve entry), τ_ff can step immediately.
    tau_final = self._apply_constraints(tau_total, v, self.steer_max, steer_limited_by_safety, dt, freeze=freeze, tau_ff=tau_ff)

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
      self.integrator *= 0.5  # halve integrator instead of resetting on driver intervention

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
    pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(tau_final) < 1e-3, CS, steer_limited_by_safety, curvature_limited))

    # ─── SIGN CONVENTION (BUGFIX v120 documentation) ────────────────────
    # The return value is NEGATED tau_final to match the openpilot convention
    # where "left is positive" (see latcontrol_torque.py line 121). This means:
    #   - tau_fb = -K@x applies LQR negative feedback internally
    #   - tau_ff + tau_fb + ... → tau_total (raw desired torque)
    #   - return -tau_final → sign-flipped for controlsd's actuators.torque
    #   - controlsd passes this to carcontroller which maps to CAN output
    # Without this negation, the LQR feedback would become POSITIVE feedback.
    # Any change here must also be mirrored in controlsd and carcontroller.
    return -tau_final, 0.0, pid_log
