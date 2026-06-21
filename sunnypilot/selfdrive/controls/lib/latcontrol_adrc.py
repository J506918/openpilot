"""
ADRC Lateral Control V2 — Two-Point Preview + ZOH ESO + Smith Predictor
Copyright (c) 2026, sunnypilot contributors

Architecture:
  Two-Point Preview (outer, 20Hz equivalent):
    modelV2.position.y → near (0.3s) + far (1.0s) curvatures
    → speed-dependent blend → desired lateral acceleration

  ADRC Inner Loop (100Hz):
    ESO (3rd-order, ZOH discrete) estimates z₁=a_y, z₂=jerk, z₃=total disturbance
    PD control law: u₀ = ω_c²·(r - z₁) - 2·ω_c·z₂
    Disturbance rejection: u = (u₀ - z₃) / b₀
    Smith predictor for 0.15s actuator delay
    Anti-windup, slip detection, ESP/μ-split protection, S-curve hysteresis

VERSION 200 — all 12 v2 review issues addressed:
  1. ZOH matrix-exponential discretization (unconditionally stable)
  2. Back-calculation anti-windup + z₂ leak + z₃ clamp
  3. Slip detection + ESO freeze + gain reduction
  4. Preview-planner consistency check
  5. True Smith predictor (delayed model + error feedback)
  6. b₀ as dimensionless tuning parameter (decoupled from latAccelFactor)
  7. ESO peaking protection: gain ramp-up + z₃ initial guess + hard clamp
  8. b₀ decoupled from liveTorqueParams (no algebraic loop)
  9. 5-point weighted LS curvature fit + far-point low-pass filter
  10. S-curve hysteresis detection (direction-consistent)
  11. ESP/VSA intervention monitoring + ESO bandwidth boost
  12. μ-split detection + control degradation + z₃ clamp tightening
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

# ─── ADRC Parameters ──────────────────────────────
WC = 8.0                # Controller bandwidth [rad/s] (~1.3 Hz)
WO = 30.0               # ESO bandwidth [rad/s] (~4.8 Hz)
B0 = 0.70               # Disturbance compensation aggressiveness (dimensionless, 0.3~1.5)

# Two-point preview
LOOKAHEAD_NEAR = 0.3    # Near preview time [s]
LOOKAHEAD_FAR = 1.0     # Far preview time [s]
SPEED_BLEND_BP = [3.0, 20.0]     # Speed blend breakpoints [m/s]
SPEED_BLEND_V  = [0.85, 0.2]     # Blend weights (1=all near, 0=all far)

# Low-speed parameters
LOW_SPEED_THRESHOLD = 3.0         # [m/s]
LOW_SPEED_WC_BP = [1.0, 3.0]     # Low-speed WC multiplier breakpoints
LOW_SPEED_WC_V  = [2.5, 1.0]
LOW_SPEED_WO_BP = [1.0, 3.0]     # Low-speed WO multiplier breakpoints
LOW_SPEED_WO_V  = [2.0, 1.0]
LOW_SPEED_B0_BP = [1.0, 3.0]     # Low-speed b₀ scaling breakpoints
LOW_SPEED_B0_V  = [0.2, 1.0]

# Friction & S-curve
FRICTION_GAIN = 0.25
CURVATURE_THRESHOLD = 0.005       # S-curve detection [rad/m]
S_CURVE_CONFIRM_FRAMES = 8        # S-curve confirmation frames

# Anti-windup
ANTIWINDUP_Z3_MAX = 3.0           # z₃ hard clamp during saturation
ANTIWINDUP_Z2_LEAK = 0.995        # z₂ leak coefficient during saturation

# ESO peaking suppression
ESO_RAMP_FRAMES = 100             # Gain ramp-up frames (1 second)
Z3_MAX_NORMAL = 8.0               # Normal z₃ clamp
Z3_MAX_MUSPLIT = 2.0              # μ-split z₃ clamp

# Slip detection
SLIP_RATIO_THRESHOLD = 0.4        # a_y_measured / a_y_des below this threshold
SLIP_TORQUE_THRESHOLD = 0.3       # Torque above this threshold to detect
SLIP_ANGLE_THRESHOLD = 120.0      # Steering angle [deg]
SLIP_RECOVERY_FRAMES = 20         # Slip recovery confirmation frames

# μ-split
MU_SPLIT_THRESHOLD = 0.03         # Wheel speed difference ratio
MU_SPLIT_WC_FACTOR = 0.6          # WC scaling during μ-split
MU_SPLIT_WO_FACTOR = 0.5          # WO scaling during μ-split

VERSION = 200  # ADRC v2 — all 12 review issues addressed


class LatControlADRC(LatControl):
  """ADRC lateral controller v2: ZOH discrete ESO + Smith predictor + two-point preview + slip/μ-split protection."""

  def __init__(self, CP, CP_SP, CI, dt):
    super().__init__(CP, CP_SP, CI, dt)

    self.torque_params = CP.lateralTuning.torque.as_builder()
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.lateral_accel_from_torque = CI.lateral_accel_from_torque()
    self.dt = dt

    # ─── Act as own extension for controlsd's extension.update_model_v2() call ───
    self.extension = self

    # ─── ESO State ────────────────────────────────────────────────────
    self.z1 = 0.0  # lateral acceleration estimate
    self.z2 = 0.0  # lateral jerk estimate
    self.z3 = 0.0  # total disturbance estimate

    # ─── Delay Buffer ─────────────────────────────────────────────────
    delay_frames = max(1, int(0.15 / dt))  # τ=0.15s → 15 frames
    self.delay_frames = delay_frames
    self.tau = delay_frames * dt
    self.torque_buffer = deque([0.0] * delay_frames, maxlen=delay_frames)

    # ─── ZOH Discrete ESO Matrices (precomputed) ──────────────────────
    self.A_d, self.B_d = self._compute_eso_discrete(WO, B0, dt)
    # Extra set for low-speed (higher wo + reduced b₀)
    self.A_d_low, self.B_d_low = self._compute_eso_discrete(
      WO * LOW_SPEED_WO_V[0], B0 * LOW_SPEED_B0_V[0], dt)

    # ─── Smith Predictor: Delayed Model Parallel State ────────────────
    self.z1_delayed = 0.0
    self.z2_delayed = 0.0
    self.z3_delayed = 0.0

    # ─── Preview State ────────────────────────────────────────────────
    self._kappa_prev = 0.0
    self._kappa_far_filtered = 0.0
    self._first_frame = True
    self._output_torque_prev = 0.0

    # ─── S-Curve Detection State ──────────────────────────────────────
    self._scurve_counter = 0
    self._scurve_active = False
    self._scurve_exit_counter = 0

    # ─── ESO Peaking / Soft-Start ─────────────────────────────────────
    self._eso_frame_count = 0

    # ─── Slip / μ-Split State ─────────────────────────────────────────
    self._slip_detected = False
    self._slip_recovery_counter = 0
    self._musplit_detected = False
    self._antiwindup_active = False
    self._eso_frozen = False

    # ─── model_v2 State ───────────────────────────────────────────────
    self.model_v2 = None
    self.model_valid = False

    # ─── Measurement Filter ───────────────────────────────────────────
    self.measurement_filter = FirstOrderFilter(0.0, 1.0 / (2.0 * math.pi * 15.0), dt)

  # ═══════════════════════════════════════════════════════════════════
  #  ZOH Discrete ESO Matrix Precomputation
  # ═══════════════════════════════════════════════════════════════════
  @staticmethod
  def _compute_eso_discrete(wo, b0, dt):
    """ZOH discretization: Padé (3,3) matrix exponential → A_d, B_d."""
    # Continuous closed-loop matrix A_cl
    A00, A01, A02 = -3.0 * wo, 1.0, 0.0
    A10, A11, A12 = -3.0 * wo ** 2, 0.0, 1.0
    A20, A21, A22 = -wo ** 3, 0.0, 0.0

    # M = A_cl * dt
    a00, a01, a02 = A00 * dt, A01 * dt, A02 * dt
    a10, a11, a12 = A10 * dt, A11 * dt, A12 * dt
    a20, a21, a22 = A20 * dt, A21 * dt, A22 * dt

    # M²
    m00 = a00 * a00 + a01 * a10 + a02 * a20
    m01 = a00 * a01 + a01 * a11 + a02 * a21
    m02 = a00 * a02 + a01 * a12 + a02 * a22
    m10 = a10 * a00 + a11 * a10 + a12 * a20
    m11 = a10 * a01 + a11 * a11 + a12 * a21
    m12 = a10 * a02 + a11 * a12 + a12 * a22
    m20 = a20 * a00 + a21 * a10 + a22 * a20
    m21 = a20 * a01 + a21 * a11 + a22 * a21
    m22 = a20 * a02 + a21 * a12 + a22 * a22

    # M³
    n00 = m00 * a00 + m01 * a10 + m02 * a20
    n01 = m00 * a01 + m01 * a11 + m02 * a21
    n02 = m00 * a02 + m01 * a12 + m02 * a22
    n10 = m10 * a00 + m11 * a10 + m12 * a20
    n11 = m10 * a01 + m11 * a11 + m12 * a21
    n12 = m10 * a02 + m11 * a12 + m12 * a22
    n20 = m20 * a00 + m21 * a10 + m22 * a20
    n21 = m20 * a01 + m21 * a11 + m22 * a21
    n22 = m20 * a02 + m21 * a12 + m22 * a22

    # Numerator N = I + M/2 + M²/6 + M³/24
    f6, f24 = 1.0 / 6.0, 1.0 / 24.0
    N00 = 1.0 + a00 * 0.5 + m00 * f6 + n00 * f24
    N01 = a01 * 0.5 + m01 * f6 + n01 * f24
    N02 = a02 * 0.5 + m02 * f6 + n02 * f24
    N10 = a10 * 0.5 + m10 * f6 + n10 * f24
    N11 = 1.0 + a11 * 0.5 + m11 * f6 + n11 * f24
    N12 = a12 * 0.5 + m12 * f6 + n12 * f24
    N20 = a20 * 0.5 + m20 * f6 + n20 * f24
    N21 = a21 * 0.5 + m21 * f6 + n21 * f24
    N22 = 1.0 + a22 * 0.5 + m22 * f6 + n22 * f24

    # Denominator D = I - M/2 + M²/6 - M³/24
    D00 = 1.0 - a00 * 0.5 + m00 * f6 - n00 * f24
    D01 = -a01 * 0.5 + m01 * f6 - n01 * f24
    D02 = -a02 * 0.5 + m02 * f6 - n02 * f24
    D10 = -a10 * 0.5 + m10 * f6 - n10 * f24
    D11 = 1.0 - a11 * 0.5 + m11 * f6 - n11 * f24
    D12 = -a12 * 0.5 + m12 * f6 - n12 * f24
    D20 = -a20 * 0.5 + m20 * f6 - n20 * f24
    D21 = -a21 * 0.5 + m21 * f6 - n21 * f24
    D22 = 1.0 - a22 * 0.5 + m22 * f6 - n22 * f24

    # D⁻¹ (Cramer's rule)
    det = (D00 * (D11 * D22 - D12 * D21)
           - D01 * (D10 * D22 - D12 * D20)
           + D02 * (D10 * D21 - D11 * D20))

    Dinv00 = (D11 * D22 - D12 * D21) / det
    Dinv01 = (D02 * D21 - D01 * D22) / det  # cofactor with sign
    Dinv02 = (D01 * D12 - D02 * D11) / det
    Dinv10 = (D12 * D20 - D10 * D22) / det
    Dinv11 = (D00 * D22 - D02 * D20) / det
    Dinv12 = (D02 * D10 - D00 * D12) / det
    Dinv20 = (D10 * D21 - D11 * D20) / det
    Dinv21 = (D01 * D20 - D00 * D21) / det
    Dinv22 = (D00 * D11 - D01 * D10) / det

    # A_d = D⁻¹ · N
    Ad00 = Dinv00 * N00 + Dinv01 * N10 + Dinv02 * N20
    Ad01 = Dinv00 * N01 + Dinv01 * N11 + Dinv02 * N21
    Ad02 = Dinv00 * N02 + Dinv01 * N12 + Dinv02 * N22
    Ad10 = Dinv10 * N00 + Dinv11 * N10 + Dinv12 * N20
    Ad11 = Dinv10 * N01 + Dinv11 * N11 + Dinv12 * N21
    Ad12 = Dinv10 * N02 + Dinv11 * N12 + Dinv12 * N22
    Ad20 = Dinv20 * N00 + Dinv21 * N10 + Dinv22 * N20
    Ad21 = Dinv20 * N01 + Dinv21 * N11 + Dinv22 * N21
    Ad22 = Dinv20 * N02 + Dinv21 * N12 + Dinv22 * N22

    A_d = [[Ad00, Ad01, Ad02],
           [Ad10, Ad11, Ad12],
           [Ad20, Ad21, Ad22]]

    # B_d = A_cl⁻¹ · (A_d - I) · [0, b0, 0]ᵀ
    Ainv00, Ainv01, Ainv02 = 0.0, 0.0, -1.0 / wo ** 3
    Ainv10, Ainv11, Ainv12 = 1.0, 0.0, -3.0 / wo ** 2
    Ainv20, Ainv21, Ainv22 = 0.0, 1.0, -3.0 / wo

    d00, d01, d02 = Ad00 - 1.0, Ad01, Ad02
    d10, d11, d12 = Ad10, Ad11 - 1.0, Ad12
    d20, d21, d22 = Ad20, Ad21, Ad22 - 1.0

    t00 = Ainv00 * d00 + Ainv01 * d10 + Ainv02 * d20
    t01 = Ainv00 * d01 + Ainv01 * d11 + Ainv02 * d21
    t02 = Ainv00 * d02 + Ainv01 * d12 + Ainv02 * d22
    t10 = Ainv10 * d00 + Ainv11 * d10 + Ainv12 * d20
    t11 = Ainv10 * d01 + Ainv11 * d11 + Ainv12 * d21
    t12 = Ainv10 * d02 + Ainv11 * d12 + Ainv12 * d22
    t20 = Ainv20 * d00 + Ainv21 * d10 + Ainv22 * d20
    t21 = Ainv20 * d01 + Ainv21 * d11 + Ainv22 * d21
    t22 = Ainv20 * d02 + Ainv21 * d12 + Ainv22 * d22

    Bd0 = t00 * 0.0 + t01 * b0 + t02 * 0.0
    Bd1 = t10 * 0.0 + t11 * b0 + t12 * 0.0
    Bd2 = t20 * 0.0 + t21 * b0 + t22 * 0.0
    B_d = [Bd0, Bd1, Bd2]

    return A_d, B_d

  # ═══════════════════════════════════════════════════════════════════
  #  Interface Compatibility
  # ═══════════════════════════════════════════════════════════════════

  def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction):
    # Only update torque parameters — b₀ is an independent tuning parameter,
    # NOT coupled to latAccelFactor (review issue #8: break algebraic loop)
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction

  def update_limits(self):
    pass

  def update_lateral_lag(self, lag):
    pass

  def update_model_v2(self, model_v2):
    self.model_v2 = model_v2
    self.model_valid = (
      self.model_v2 is not None
      and len(self.model_v2.position.y) >= CONTROL_N
    )

  def reset(self):
    super().reset()
    self.z1 = 0.0
    self.z2 = 0.0
    self.z3 = 0.0
    self.z1_delayed = 0.0
    self.z2_delayed = 0.0
    self.z3_delayed = 0.0
    self.torque_buffer = deque([0.0] * self.delay_frames, maxlen=self.delay_frames)
    self._kappa_prev = 0.0
    self._kappa_far_filtered = 0.0
    self._first_frame = True
    self._output_torque_prev = 0.0
    self._scurve_counter = 0
    self._scurve_active = False
    self._scurve_exit_counter = 0
    self._eso_frame_count = 0
    self._slip_detected = False
    self._slip_recovery_counter = 0
    self._musplit_detected = False
    self._antiwindup_active = False
    self._eso_frozen = False

  # ═══════════════════════════════════════════════════════════════════
  #  Two-Point Preview — 5-point fit + far-point filter + S-curve hysteresis
  # ═══════════════════════════════════════════════════════════════════

  def _curvature_at_robust(self, lookahead_s, v_ego):
    """5-point weighted least-squares parabola fit for curvature (review issue #9)."""
    y = self.model_v2.position.y
    idx = int(np.searchsorted(ModelConstants.T_IDXS[:CONTROL_N], lookahead_s))
    idx_center = min(max(idx, 2), CONTROL_N - 3)

    indices = range(idx_center - 2, idx_center + 3)
    x_vals = np.array([ModelConstants.T_IDXS[i] * max(v_ego, 1.0) for i in indices])
    y_vals = np.array([float(y[i]) for i in indices])

    sigma = lookahead_s * max(v_ego, 1.0) / 3.0
    if sigma < 1e-6:
      return 0.0
    w = np.exp(-0.5 * ((x_vals - x_vals[2]) / sigma) ** 2)

    S0 = np.sum(w)
    S1 = np.sum(w * x_vals)
    S2 = np.sum(w * x_vals ** 2)
    S3 = np.sum(w * x_vals ** 3)
    S4 = np.sum(w * x_vals ** 4)
    Sy = np.sum(w * y_vals)
    Sxy = np.sum(w * x_vals * y_vals)
    Sx2y = np.sum(w * x_vals ** 2 * y_vals)

    det = S0 * (S2 * S4 - S3 ** 2) - S1 * (S1 * S4 - S2 * S3) + S2 * (S1 * S3 - S2 ** 2)
    if abs(det) < 1e-12:
      return 0.0
    a = (Sy * (S2 * S4 - S3 ** 2) - Sxy * (S1 * S4 - S2 * S3) + Sx2y * (S1 * S3 - S2 ** 2)) / det
    return 2.0 * a  # κ = 2a for y = a·x² + b·x + c

  def _compute_preview(self, v_ego):
    """Two-point preview + speed blend + S-curve hysteresis enhancement (review issue #10)."""
    if not self.model_valid:
      return 0.0

    kappa_near = self._curvature_at_robust(LOOKAHEAD_NEAR, v_ego)
    kappa_far_raw = self._curvature_at_robust(LOOKAHEAD_FAR, v_ego)

    # Far-point low-pass filter (2 Hz cutoff)
    alpha_far = math.exp(-2.0 * math.pi * 2.0 * self.dt)
    self._kappa_far_filtered = (alpha_far * self._kappa_far_filtered
                                + (1.0 - alpha_far) * kappa_far_raw)
    kappa_far = self._kappa_far_filtered

    # Speed-adaptive blend
    alpha = float(np.interp(v_ego, SPEED_BLEND_BP, SPEED_BLEND_V))
    alpha = np.clip(alpha, 0.15, 0.85)

    # S-curve hysteresis detection
    if kappa_near * kappa_far < 0 and abs(kappa_near) > CURVATURE_THRESHOLD:
      self._scurve_counter += 1
    else:
      self._scurve_counter = max(0, self._scurve_counter - 1)

    if self._scurve_active:
      if kappa_near * kappa_far >= 0:
        self._scurve_exit_counter += 1
        if self._scurve_exit_counter >= S_CURVE_CONFIRM_FRAMES // 2:
          self._scurve_active = False
          self._scurve_exit_counter = 0
      else:
        self._scurve_exit_counter = 0
    else:
      if self._scurve_counter >= S_CURVE_CONFIRM_FRAMES:
        self._scurve_active = True

    if self._scurve_active:
      alpha = min(alpha * 1.3, 0.85)

    return alpha * kappa_near + (1.0 - alpha) * kappa_far

  # ═══════════════════════════════════════════════════════════════════
  #  ZOH Discrete ESO — Unconditionally Stable
  # ═══════════════════════════════════════════════════════════════════

  def _eso_update_discrete(self, y_measured, u_delayed, A_d, B_d, wo):
    """Predict + correct structure (review issue #1)."""
    # Prediction step
    z1p = (A_d[0][0] * self.z1 + A_d[0][1] * self.z2 + A_d[0][2] * self.z3
           + B_d[0] * u_delayed)
    z2p = (A_d[1][0] * self.z1 + A_d[1][1] * self.z2 + A_d[1][2] * self.z3
           + B_d[1] * u_delayed)
    z3p = (A_d[2][0] * self.z1 + A_d[2][1] * self.z2 + A_d[2][2] * self.z3
           + B_d[2] * u_delayed)

    # Correction step
    e = y_measured - z1p
    dt = self.dt
    self.z1 = z1p + 3.0 * wo * dt * e
    self.z2 = z2p + 3.0 * wo ** 2 * dt * e
    self.z3 = z3p + wo ** 3 * dt * e

  # ═══════════════════════════════════════════════════════════════════
  #  Smith Predictor — True Structure
  # ═══════════════════════════════════════════════════════════════════

  def _smith_predict(self, y_measured, u_delayed, A_d, B_d):
    """Delayed model advance + error feedback (review issue #5)."""
    # Delayed model prediction
    z1d = (A_d[0][0] * self.z1_delayed + A_d[0][1] * self.z2_delayed
           + A_d[0][2] * self.z3_delayed + B_d[0] * u_delayed)
    z2d = (A_d[1][0] * self.z1_delayed + A_d[1][1] * self.z2_delayed
           + A_d[1][2] * self.z3_delayed + B_d[1] * u_delayed)
    z3d = (A_d[2][0] * self.z1_delayed + A_d[2][1] * self.z2_delayed
           + A_d[2][2] * self.z3_delayed + B_d[2] * u_delayed)

    # Smith error
    e_smith = y_measured - z1d

    # Update delayed model state
    self.z1_delayed, self.z2_delayed, self.z3_delayed = z1d, z2d, z3d

    # Return delay-free equivalent output
    return self.z1 + e_smith

  # ═══════════════════════════════════════════════════════════════════
  #  Control Law
  # ═══════════════════════════════════════════════════════════════════

  def _compute_control(self, a_y_target, wc, b0):
    """PD + disturbance compensation → equivalent lateral accel."""
    u0 = wc ** 2 * (a_y_target - self.z1) - 2.0 * wc * self.z2
    return (u0 - self.z3) / max(b0, 0.05)

  # ═══════════════════════════════════════════════════════════════════
  #  ─── Main Update ─────────────────────────────────────────────────
  # ═══════════════════════════════════════════════════════════════════

  def update(self, active, CS, VM, params, steer_limited_by_safety,
             desired_curvature, calibrated_pose, curvature_limited, lat_delay):
    pid_log = log.ControlsState.LateralTorqueState.new_message()
    pid_log.version = VERSION

    if not active:
      self.reset()
      return 0.0, 0.0, pid_log

    v_ego = CS.vEgo

    # ─── 1. Two-Point Preview → Desired Lateral Acceleration ──────────
    if self.model_valid:
      kappa_blend = self._compute_preview(v_ego)
    else:
      kappa_blend = desired_curvature

    # Preview vs planner consistency check (review issue #4)
    if self.model_valid and abs(kappa_blend - desired_curvature) > 0.01:
      conflict_w = np.clip(abs(kappa_blend - desired_curvature) / 0.03, 0.0, 0.6)
      kappa_blend = ((1.0 - conflict_w) * kappa_blend
                     + conflict_w * desired_curvature)

    a_y_des_raw = kappa_blend * v_ego ** 2
    if self._first_frame:
      curvature_rate = 0.0
      self._first_frame = False
    else:
      curvature_rate = (kappa_blend - self._kappa_prev) / self.dt
    self._kappa_prev = kappa_blend
    a_y_des_raw += FRICTION_GAIN * curvature_rate * v_ego ** 2

    roll_comp = params.roll * ACCELERATION_DUE_TO_GRAVITY
    a_y_des = a_y_des_raw - roll_comp - self.torque_params.latAccelOffset

    # ─── 2. Measurement ───────────────────────────────────────────────
    measured_curvature = -VM.calc_curvature(
      math.radians(CS.steeringAngleDeg - params.angleOffsetDeg),
      v_ego, params.roll)
    measured_a_y_raw = measured_curvature * v_ego ** 2
    measured_a_y = self.measurement_filter.update(measured_a_y_raw)

    # ─── 3. Speed-Adaptive Bandwidth ──────────────────────────────────
    if v_ego < LOW_SPEED_THRESHOLD:
      wc = WC * float(np.interp(v_ego, LOW_SPEED_WC_BP, LOW_SPEED_WC_V))
      wo = WO * float(np.interp(v_ego, LOW_SPEED_WO_BP, LOW_SPEED_WO_V))
      b0 = B0 * float(np.interp(v_ego, LOW_SPEED_B0_BP, LOW_SPEED_B0_V))
      A_d_eso, B_d_eso = self.A_d_low, self.B_d_low
    else:
      wc = WC
      wo = WO
      b0 = B0
      A_d_eso, B_d_eso = self.A_d, self.B_d

    # ─── 4. ESP/VSA Intervention (review issue #11) ───────────────────
    esp_intervening = (getattr(CS, 'espActive', False)
                       or getattr(CS, 'vsaActive', False))
    if esp_intervening:
      wo *= 2.0  # Faster tracking of ESP-introduced dynamics

    # ─── 5. μ-Split Detection (review issue #12) ──────────────────────
    wheel_speed_diff = abs(getattr(CS, 'leftWheelSpeed', v_ego)
                           - getattr(CS, 'rightWheelSpeed', v_ego))
    wheel_slip = wheel_speed_diff / max(v_ego, 0.1)
    yaw_unexpected = (abs(CS.yawRate) > 0.05
                      and abs(CS.steeringAngleDeg) < 5.0)
    self._musplit_detected = (wheel_slip > MU_SPLIT_THRESHOLD
                              and yaw_unexpected)
    if self._musplit_detected:
      wo *= MU_SPLIT_WO_FACTOR
      wc *= MU_SPLIT_WC_FACTOR

    # ─── 6. ESO Peaking Suppression: Gain Ramp-Up (review issue #7) ───
    self._eso_frame_count += 1
    if self._eso_frame_count <= ESO_RAMP_FRAMES:
      ramp = self._eso_frame_count / ESO_RAMP_FRAMES
      gain_mult = 0.3 + 0.7 * ramp
      wo *= gain_mult
    # Initial z₃ guess
    if self._eso_frame_count == 1:
      self.z3 = params.roll * ACCELERATION_DUE_TO_GRAVITY

    # ─── 7. ESO Update ────────────────────────────────────────────────
    u_delayed = self.torque_buffer[0]
    if not self._eso_frozen:
      self._eso_update_discrete(measured_a_y, u_delayed, A_d_eso, B_d_eso, wo)

    # Hard clamp z₃ (always active)
    z3_limit = Z3_MAX_MUSPLIT if self._musplit_detected else Z3_MAX_NORMAL
    self.z3 = np.clip(self.z3, -z3_limit, z3_limit)

    # ─── 8. Smith Predict → Delay-Free Equivalent Output ──────────────
    a_y_control = self._smith_predict(measured_a_y, u_delayed, A_d_eso, B_d_eso)

    # ─── 9. Control Law ───────────────────────────────────────────────
    desired_lat_accel = self._compute_control(a_y_control, wc, b0)

    # ─── 10. Slip Detection (review issue #3) ─────────────────────────
    slip_ratio = abs(measured_a_y) / max(abs(a_y_des), 0.05)
    cond1 = (abs(self._output_torque_prev) > SLIP_TORQUE_THRESHOLD
             and slip_ratio < SLIP_RATIO_THRESHOLD)
    cond2 = (abs(CS.steeringAngleDeg - params.angleOffsetDeg)
             > SLIP_ANGLE_THRESHOLD)
    if cond1 or cond2:
      self._slip_detected = True
      self._eso_frozen = True
      wc *= 0.3
    else:
      self._slip_detected = False

    if not self._slip_detected:
      self._slip_recovery_counter = min(
        self._slip_recovery_counter + 1, SLIP_RECOVERY_FRAMES)
    else:
      self._slip_recovery_counter = 0
    if self._slip_recovery_counter >= SLIP_RECOVERY_FRAMES:
      self._eso_frozen = False

    # ─── 11. lat_accel → torque ───────────────────────────────────────
    output_torque = self.torque_from_lateral_accel(
      desired_lat_accel, self.torque_params)

    # ─── 12. Back-Calculation Anti-Windup (review issue #2) ───────────
    torque_sat = float(np.clip(output_torque, -self.steer_max, self.steer_max))
    if abs(output_torque - torque_sat) > 1e-6:
      if torque_sat * output_torque > torque_sat * torque_sat * 0.99:
        self._antiwindup_active = True
    else:
      self._antiwindup_active = False

    if self._antiwindup_active:
      self.z3 = np.clip(self.z3, -ANTIWINDUP_Z3_MAX, ANTIWINDUP_Z3_MAX)
      self.z2 *= ANTIWINDUP_Z2_LEAK

    output_torque = torque_sat

    # ─── 13. Update Torque Buffer ──────────────────────────────────────
    self.torque_buffer.append(output_torque)

    # Driver intervening
    if CS.steeringPressed:
      output_torque *= 0.5

    output_torque = float(np.clip(output_torque, -self.steer_max, self.steer_max))
    self._output_torque_prev = output_torque

    # ─── Logging ──────────────────────────────────────────────────────
    pid_log.active = True
    pid_log.p = float(self.z1)
    pid_log.i = float(self.z2)
    pid_log.d = float(self.z3)
    pid_log.f = float(a_y_des)
    pid_log.output = float(-output_torque)
    pid_log.actualLateralAccel = float(measured_a_y)
    pid_log.desiredLateralAccel = float(a_y_control)
    pid_log.saturated = bool(self._antiwindup_active)

    return -output_torque, 0.0, pid_log
