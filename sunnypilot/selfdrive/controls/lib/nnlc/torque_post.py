"""
Directional Inertia Post-Processor — "不再左右摇摆"

Philosophy: human steering is a series of SMOOTH, SINGLE-DIRECTION movements.
Once you start correcting right, you stay right until the correction is done.
No oscillation. No micro-jitter.

Three mechanisms:
  1. Hysteresis deadband  — entry threshold ≠ exit threshold
  2. Direction lock        — once direction chosen, hold it until error resolved
  3. Return-to-zero ramp   — smooth release, not snap-back
"""
import numpy as np
from collections import deque


class DirectionalInertia:
  """
  Hysteresis-based torque direction manager.
  Prevents left-right-left-right oscillation.
  """

  def __init__(self, dt=0.01):
    self.dt = dt

    # ── Hysteresis deadband ──
    # When centered:  wide threshold to START a correction
    # When correcting: narrow threshold to FINISH
    self.deadband_enter = 0.15   # must exceed this to START correcting (meters)
    self.deadband_exit = 0.05    # must go below this to STOP correcting

    # ── State ──
    self._correcting = False     # are we in an active correction?
    self._correction_dir = 0     # -1 (left), 0 (none), +1 (right)
    self._correction_ramp = 0.0  # 0→1 ramp for smooth entry
    self._target_torque = 0.0    # target torque for this correction
    self._torque_at_start = 0.0  # torque when correction began
    self._frames_in_correction = 0
    self._frames_in_deadband = 0
    self._output = 0.0

    # ── Rate limiting ──
    self.max_rate = 2.5          # Nm/s (human: slow and deliberate)
    self.max_per_frame = self.max_rate * dt
    self.rate_exit = 1.5         # even slower for return-to-center
    self.max_per_frame_exit = self.rate_exit * dt

    # ── Smoothing ──
    self.ema_alpha = 0.25        # smooth but responsive
    self._ema = 0.0

  def reset(self):
    self._correcting = False
    self._correction_dir = 0
    self._correction_ramp = 0.0
    self._target_torque = 0.0
    self._torque_at_start = 0.0
    self._frames_in_correction = 0
    self._frames_in_deadband = 0
    self._output = 0.0
    self._ema = 0.0

  def process(self, desired_torque, lateral_error, steering_pressed, active):
    """
    Returns torque with directional inertia applied.
    """
    if not active or steering_pressed:
      self.reset()
      return 0.0

    abs_err = abs(lateral_error)
    err_sign = 1 if lateral_error > 0 else (-1 if lateral_error < 0 else 0)

    # ── Decision: start, continue, or end correction ──
    if not self._correcting:
      # Currently in deadband. Check if we should start correcting.
      if abs_err > self.deadband_enter:
        # Start a new correction
        self._correcting = True
        self._correction_dir = -err_sign  # opposite of error direction
        self._correction_ramp = 0.0
        self._target_torque = desired_torque
        self._torque_at_start = self._output
        self._frames_in_correction = 0
      else:
        # Stay in deadband — HOLD current torque (freeze!)
        # Gradual decay toward zero, but very slowly
        self._frames_in_deadband += 1
        decay = max(0.0, 1.0 - self._frames_in_deadband / 200.0)  # 2s to zero
        self._output *= decay
        # If output is very small, just zero it
        if abs(self._output) < 0.05:
          self._output = 0.0
        return self._output

    # ── Currently correcting ──
    self._frames_in_correction += 1
    self._frames_in_deadband = 0

    # Check if correction should end
    should_end = (
      abs_err < self.deadband_exit or  # error is small enough
      self._frames_in_correction > 150  # max 1.5s correction (safety)
    )

    # Check if desired_torque has reversed direction mid-correction
    # If so, ignore the reversal — we commit to the direction we started
    desired_dir = 1 if desired_torque > 0.02 else (-1 if desired_torque < -0.02 else 0)

    if should_end:
      # End correction — smooth return to zero
      self._correcting = False
      self._correction_dir = 0
      self._correction_ramp = 0.0
      # Don't snap to zero — let EMA handle the decay
      torque = desired_torque * 0.3  # gentle final touch
    elif desired_dir != 0 and desired_dir != self._correction_dir:
      # Desired torque wants to reverse direction mid-correction
      # → IGNORE the reversal. Hold current torque or reduce magnitude.
      # Only reduce magnitude, never reverse sign
      if self._output * desired_torque < 0:
        # Signs differ — desired wants opposite direction
        # Gradually reduce current output but don't reverse
        reduction = max(0.0, 1.0 - 5.0 / 100.0)  # rapid decay when fighting
        torque = self._output * reduction
      else:
        torque = desired_torque
    else:
      # Normal correction — ramp toward target
      self._correction_ramp = min(1.0, self._correction_ramp + 0.04)  # ~400ms ramp
      # Blend: from start torque toward target
      raw = self._torque_at_start * (1.0 - self._correction_ramp) + desired_torque * self._correction_ramp
      # Ensure direction consistency
      if self._correction_dir == 1:
        torque = max(0.0, raw)  # right correction: never go negative
      elif self._correction_dir == -1:
        torque = min(0.0, raw)  # left correction: never go positive
      else:
        torque = raw

    # ── Rate limit ──
    rate = self.max_per_frame_exit if should_end else self.max_per_frame
    delta = torque - self._output
    if abs(delta) > rate:
      delta = np.sign(delta) * rate
      torque = self._output + delta

    # ── EMA smooth ──
    self._output = self._output * (1.0 - self.ema_alpha) + torque * self.ema_alpha

    return self._output


class TorquePostProcessor:
  """
  Human-like torque shaping with directional inertia.
  Wraps DirectionalInertia with caster trail and safety limits.
  """

  def __init__(self, max_torque=4.0, dt=0.01):
    self.max_torque = max_torque
    self.dt = dt

    # ── Directional inertia (core) ──
    self.inertia = DirectionalInertia(dt=dt)

    # ── Caster trail ──
    self.caster_coeff = 0.05
    self.caster_max = 1.2

    # ── State ──
    self._curve_exiting = False
    self._exit_counter = 0
    self._active = False

  def reset(self):
    self.inertia.reset()
    self._curve_exiting = False
    self._exit_counter = 0
    self._active = False

  def process(self, raw_torque, lateral_error, actual_lateral_accel,
              desired_curvature, actual_curvature, v_ego, steering_pressed, active):
    """
    Full post-processing pipeline.
    """
    if not active:
      self.reset()
      return 0.0

    self._active = True

    # ── 1. Safety hard clamp ──
    torque = float(np.clip(raw_torque, -self.max_torque, self.max_torque))

    # ── 2. Caster trail during curve exit ──
    des_k_abs = abs(float(desired_curvature))
    act_k_abs = abs(float(actual_curvature))
    is_exiting = (des_k_abs < act_k_abs * 0.6 and act_k_abs > 0.001 and v_ego > 3.0)

    if is_exiting and not self._curve_exiting:
      self._curve_exiting = True
      self._exit_counter = 0

    if self._curve_exiting:
      self._exit_counter += 1
      lat_acc = float(actual_lateral_accel)
      if abs(lat_acc) > 0.05:
        caster = -np.sign(lat_acc) * min(abs(lat_acc) * self.caster_coeff, self.caster_max)
        blend = min(1.0, self._exit_counter / 30.0)
        torque = torque * (1.0 - blend * 0.35) + caster * blend * 0.35

      if act_k_abs < 0.0005 or self._exit_counter > 100:
        self._curve_exiting = False
        self._exit_counter = 0

    # ── 3. Directional inertia (the main protection against oscillation) ──
    torque = self.inertia.process(torque, lateral_error, steering_pressed, active)

    return torque

  @property
  def is_correcting(self):
    return self.inertia._correcting
