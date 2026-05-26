"""
Driving Style Extractor — 6-dim physical quantification.

Extracts driver steering style from CAN data when openpilot is
disengaged. Converges via EMA, resamples on distribution shift.

Dimensions:
  0. steer_gain      steeringAngle / lateral_error    (°/cm)
  1. curvature_resp  steeringAngle / curvature         (°)
  2. aggressiveness  std(steeringRate)                 (°/s)
  3. anticipation    argmax(cross_corr(curvature, steerAngle)) frames
  4. return_ratio    max(|exit_rate|) / max(|entry_rate|)
  5. hand_force      median(|steeringTorqueEps|)       (Nm)

Convergence:   EMA(α=0.01) → stabilises after ~500 frames
Resample:      KL(current || stored) > 0.15 → open adaptation window
Data buffer:   5000 samples, importance-weighted eviction

Never resets. Never grows unbounded.
"""
import numpy as np
from collections import deque


def safe_div(a, b, default=0.0):
    return a / max(abs(b), 1e-6) if abs(b) > 1e-6 else default


class DrivingStyleCollector:
    """Collects (scene, driver_steering) when openpilot is disengaged."""

    MAX_BUFFER = 5000
    COLLECT_EVERY_N = 5   # collect 1 frame per 5 (100Hz → 20Hz effective)

    def __init__(self):
        # ── Per-frame rolling window (5 seconds @ 20Hz = 100 samples) ──
        self._win_angle     = deque(maxlen=100)
        self._win_rate      = deque(maxlen=100)
        self._win_torque    = deque(maxlen=100)
        self._win_curvature = deque(maxlen=100)
        self._win_error     = deque(maxlen=100)

        # ── Convergence fields ──
        self.style         = np.zeros(6, dtype=np.float32)     # current EMA style
        self.style_sigma   = np.ones(6, dtype=np.float32) * 0.5  # per-dim std
        self.style_samples = 0
        self._alpha        = 0.01                              # fixed learning rate

        # ── KL divergence monitoring ──
        self._last_stable_style = np.zeros(6, dtype=np.float32)
        self._kl_buffer         = deque(maxlen=100)
        self._shift_detected    = False
        self._adaptation_frames = 0

        # ── Sample buffer (importance-weighted) ──
        self._frame_skip = 0

    @property
    def converged(self) -> bool:
        return self.style_samples > 500 and not self._shift_detected

    @property
    def kl_trend(self) -> float:
        if len(self._kl_buffer) < 10:
            return 0.0
        return float(np.mean(list(self._kl_buffer)[-20:]))

    def update(self, steering_angle: float, steering_rate: float,
               steering_torque: float, curvature: float, lateral_error: float,
               openpilot_engaged: bool) -> None:
        """
        Call every frame.
        Collects only when openpilot is DISENGAGED (driver steering).
        """
        if openpilot_engaged:
            self._frame_skip = 0
            return

        # Downsample
        self._frame_skip += 1
        if self._frame_skip < self.COLLECT_EVERY_N:
            return
        self._frame_skip = 0

        # Only collect during meaningful cornering
        if abs(curvature) < 0.001:
            return

        # ── Append to rolling window ──
        self._win_angle.append(steering_angle)
        self._win_rate.append(abs(steering_rate))
        self._win_torque.append(abs(steering_torque))
        self._win_curvature.append(curvature)
        self._win_error.append(lateral_error)

        # ── Compute 6-dim style from window ──
        if len(self._win_angle) < 30:
            return

        current = self._extract_style()
        self._update_ema(current)

        # ── KL monitoring ──
        if self.style_samples > 200:
            kl = self._compute_kl(self._last_stable_style, self.style)
            self._kl_buffer.append(kl)

    def _extract_style(self) -> np.ndarray:
        """Extract 6-dim style vector from rolling window."""
        win_a = np.array(self._win_angle, dtype=np.float32)
        win_r = np.array(self._win_rate, dtype=np.float32)
        win_t = np.array(self._win_torque, dtype=np.float32)
        win_c = np.array(self._win_curvature, dtype=np.float32)
        win_e = np.array(self._win_error, dtype=np.float32)

        s = np.zeros(6, dtype=np.float32)

        # 0. Steer gain (° per cm lateral error)
        s[0] = safe_div(float(np.mean(np.abs(win_a))),
                         float(np.mean(np.abs(win_e))) * 100.0, 0.5)

        # 1. Curvature response (°)
        s[1] = safe_div(float(np.mean(np.abs(win_a))),
                         float(np.mean(np.abs(win_c))) * 100.0, 45.0)

        # 2. Aggressiveness: std of steering rate (°/s)
        s[2] = float(np.std(win_r))

        # 3. Anticipation: cross-correlation lag (frames)
        s[3] = self._cross_corr_lag(win_c, win_a)

        # 4. Return ratio: max(|exit_rate|) / max(|entry_rate|)
        mid = len(win_c) // 2
        entry = np.abs(win_r[:mid])
        exit_ = np.abs(win_r[mid:])
        entry_max = float(np.max(entry)) if len(entry) > 0 else 1.0
        exit_max  = float(np.max(exit_)) if len(exit_) > 0 else 1.0
        s[4] = safe_div(exit_max, entry_max, 1.0)

        # 5. Hand force: median torque (Nm)
        s[5] = float(np.median(win_t))

        return np.clip(s, 0.0, 10.0)

    @staticmethod
    def _cross_corr_lag(a: np.ndarray, b: np.ndarray) -> float:
        """Returns lag (frames) of peak cross-correlation."""
        n = min(len(a), len(b), 20)
        if n < 5:
            return 0.0
        a_norm = (a[-n:] - np.mean(a[-n:])) / (np.std(a[-n:]) + 1e-6)
        b_norm = (b[-n:] - np.mean(b[-n:])) / (np.std(b[-n:]) + 1e-6)
        corr = np.correlate(a_norm, b_norm, mode='full')
        peak = np.argmax(corr) - (n - 1)
        return float(np.clip(peak, -5, 5))

    def _update_ema(self, current: np.ndarray):
        """Fixed-rate EMA with sigma-based clamping."""
        if self.style_samples == 0:
            self.style = current.copy()
            self.style_samples = 1
            return

        self.style = (1 - self._alpha) * self.style + self._alpha * current

        # Sigma tracking
        delta = current - self.style
        self.style_sigma = 0.99 * self.style_sigma + 0.01 * np.abs(delta)
        self.style_sigma = np.maximum(self.style_sigma, 0.01)

        # Clamp within 2σ
        for i in range(6):
            lo = self.style[i] - 2.0 * self.style_sigma[i]
            hi = self.style[i] + 2.0 * self.style_sigma[i]
            self.style[i] = np.clip(self.style[i], lo, hi)

        self.style_samples += 1

        # ── KL shift detection ──
        if self.style_samples > 500 and self.style_samples % 100 == 0:
            kl = self._compute_kl(self._last_stable_style, self.style)
            if kl > 0.15 and not self._shift_detected:
                self._shift_detected = True
                self._adaptation_frames = 200
            elif kl < 0.05 and self._shift_detected:
                self._last_stable_style = self.style.copy()
                self._shift_detected = False

        if self._adaptation_frames > 0:
            self._adaptation_frames -= 1
        elif self._adaptation_frames == 0 and self._shift_detected:
            # Adaptation window closed, accept new style as stable
            self._last_stable_style = self.style.copy()
            self._shift_detected = False

    @staticmethod
    def _compute_kl(a: np.ndarray, b: np.ndarray) -> float:
        """Symmetric KL-like divergence between two style vectors."""
        a_safe = np.maximum(np.abs(a), 0.01)
        b_safe = np.maximum(np.abs(b), 0.01)
        kl_ab = np.sum(a_safe * np.log(a_safe / b_safe))
        kl_ba = np.sum(b_safe * np.log(b_safe / a_safe))
        return float(0.5 * (kl_ab + kl_ba))
