"""
Online training engine — steering angle target, KL-triggered.

Target:        driver steering angle (not torque delta)
Trigger:       KL(style_current || model_output_distribution) > threshold
Convergence:   loss plateau → stop training, only infer
Resample:      importance-weighted eviction (keep most informative)
Style shift:   temporarily increase learning rate (0.0001→0.0005), never reset

Sample quality:
  1. Minimum cornering curvature (>0.001) to filter straight driving
  2. Speed filter (>2.0 m/s, not parking)
  3. IQR-based outlier rejection per batch
  4. Duplicate avoidance (features cosine similarity > 0.98 → skip)
  5. Importance = |prediction_error| (keep samples where model struggles)
"""
import numpy as np
from collections import deque


class OnlineTrainer:
    """
    Manages online learning from driver steering samples.

    States: COLLECTING → TRAINING → CONVERGED → MONITORING
    """

    def __init__(self, model, buffer_size=5000, batch_size=64,
                 train_interval_frames=300, min_samples=100):
        self.model = model
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.train_interval_frames = train_interval_frames
        self.min_samples = min_samples

        # ── Sample buffer (importance-weighted) ──
        self._buffer = []
        self._frame_counter = 0
        self._total_samples = 0
        self._total_train_steps = 0
        self._last_train_loss = None

        # ── Lifelong learning rate (converges, never resets) ──
        self._base_lr = 0.001
        self._lr_min = 0.0001
        self._lr = self._base_lr
        self._lr_decay_steps = 50000
        self._style_shift_boost = False
        self._boost_remaining = 0

        # ── Training state ──
        self.converged = False
        self._convergence_counter = 0
        self._convergence_threshold = 10  # consecutive epochs with no improvement

        # ── Quality metrics ──
        self._recent_losses = deque(maxlen=100)
        self._recent_errors = deque(maxlen=200)
        self._rejected_samples = 0
        self._accepted_samples = 0
        self._duplicate_rejects = 0

        # ── Outlier bounds (IQR-based) ──
        self._error_q1 = 0.5
        self._error_q3 = 3.0

        # ── KL divergence tracking ──
        self._pred_error_ema = 0.0
        self._kl_baseline = None

    # ── Public API ──

    def add_sample(self, features: np.ndarray, driver_steering_angle: float,
                   system_steering_angle: float, v_ego: float,
                   curvature: float) -> bool:
        """
        Add a driving sample. Returns True if accepted.
        Only collects during cornering (|curvature| > 0.001).
        """
        # Basic filters
        if v_ego < 2.0:
            return False
        if abs(curvature) < 0.001:
            return False
        angle_diff = abs(driver_steering_angle - system_steering_angle)
        if angle_diff < 0.5:
            return False  # too close to system → not informative

        # Duplicate check
        if self._is_duplicate(features):
            self._duplicate_rejects += 1
            return False

        # Importance = how much model would disagree
        importance = self._estimate_importance(features, driver_steering_angle)

        sample = {
            'features': features.copy(),
            'target': float(driver_steering_angle),
            'importance': importance,
            'frame': self._frame_counter,
            'v_ego': float(v_ego),
            'curvature': float(curvature),
        }

        self._insert_sorted(sample)
        self._total_samples += 1
        self._accepted_samples += 1

        # Update error distribution
        self._recent_errors.append(importance)
        if len(self._recent_errors) >= 20:
            sorted_e = sorted(self._recent_errors)
            n = len(sorted_e)
            self._error_q1 = sorted_e[n // 4]
            self._error_q3 = sorted_e[3 * n // 4]

        return True

    def update(self, style_collector=None):
        """
        Called every frame.
        Returns (should_train: bool, train_loss: float or None).
        Training triggered by KL divergence, not fixed interval.
        """
        self._frame_counter += 1

        should_train = False
        train_loss = None

        # ── KL-based training trigger ──
        kl = 0.0
        if style_collector is not None:
            kl = style_collector.kl_trend

        trigger = (
            kl > 0.08 and                                      # style vs model mismatch
            len(self._buffer) >= self.min_samples and          # enough data
            self._frame_counter % 50 == 0                      # throttle
        )

        # Also periodic training when not converged
        if not self.converged and self._frame_counter % self.train_interval_frames == 0:
            trigger = True

        if trigger and len(self._buffer) >= self.min_samples:
            self._update_learning_rate()
            if style_collector is not None:
                self._handle_style_shift(style_collector)
            train_loss = self._train_batch()
            self._total_train_steps += 1
            should_train = True

            if train_loss is not None:
                self._recent_losses.append(train_loss)
                self._last_train_loss = train_loss
                self._check_convergence(train_loss)

            self._evict_low_importance()

        return should_train, train_loss

    # ── Training ──

    def _train_batch(self):
        """Importance-weighted batch training with outlier rejection."""
        n_available = len(self._buffer)
        if n_available < self.batch_size:
            return None

        # Importance-weighted sampling
        importances = np.array([s['importance'] for s in self._buffer])
        importances = np.maximum(importances, 0.01)
        importances /= importances.sum()

        indices = np.random.choice(n_available, self.batch_size, p=importances, replace=True)

        # Outlier rejection
        iqr = max(self._error_q3 - self._error_q1, 0.1)
        upper = self._error_q3 + 2.5 * iqr

        errors = []
        for i in indices:
            s = self._buffer[i]
            if s['importance'] > upper:
                continue
            err = self.model.learn_personalization(s['features'], s['target'])
            if err is not None and not np.isnan(err):
                errors.append(abs(err))

        if not errors:
            return None
        return float(np.mean(errors))

    def _check_convergence(self, current_loss: float):
        """Detect loss plateau → stop training."""
        if len(self._recent_losses) < 20:
            return

        recent = list(self._recent_losses)[-20:]
        recent_mean = np.mean(recent)
        loss_std = np.std(recent)

        # Low variance + no improvement → converged
        if loss_std < max(recent_mean * 0.05, 1e-5):
            self._convergence_counter += 1
        else:
            self._convergence_counter = max(0, self._convergence_counter - 1)

        if self._convergence_counter >= self._convergence_threshold:
            self.converged = True

    # ── Learning rate ──

    def _update_learning_rate(self):
        """Cosine decay to min, style shift gives temporary boost."""
        if self._boost_remaining > 0:
            self._lr = 0.0005  # boosted
            self._boost_remaining -= 1
        else:
            progress = min(1.0, self._total_train_steps / self._lr_decay_steps)
            self._lr = self._lr_min + 0.5 * (self._base_lr - self._lr_min) * (1.0 + np.cos(np.pi * progress))
        self.model._lr_personalization = self._lr

    def _handle_style_shift(self, style_collector):
        """Temporary lr boost on style shift, no reset."""
        if style_collector._shift_detected and not self._style_shift_boost:
            self._style_shift_boost = True
            self._boost_remaining = 200  # ~200 training steps
        elif not style_collector._shift_detected:
            self._style_shift_boost = False

    # ── Buffer management ──

    def _insert_sorted(self, sample):
        """Insert sample, maintaining max size by evicting lowest importance."""
        self._buffer.append(sample)
        if len(self._buffer) > self.buffer_size:
            self._evict_one()

    def _evict_one(self):
        """Evict the sample with lowest importance."""
        if not self._buffer:
            return
        min_idx = min(range(len(self._buffer)),
                      key=lambda i: self._buffer[i]['importance'])
        del self._buffer[min_idx]

    def _evict_low_importance(self):
        """Periodic cleanup: evict bottom 5% by importance."""
        if len(self._buffer) < self.buffer_size * 0.8:
            return
        n_evict = max(1, len(self._buffer) // 20)
        sorted_idx = sorted(range(len(self._buffer)),
                            key=lambda i: self._buffer[i]['importance'])
        for i in sorted(sorted_idx[:n_evict], reverse=True):
            del self._buffer[i]

    def _is_duplicate(self, features: np.ndarray) -> bool:
        """Check if features are too similar to any existing sample."""
        if len(self._buffer) < 10:
            return False
        # Check last 50 samples for cosine similarity
        recent = [s['features'] for s in self._buffer[-50:]]
        if not recent:
            return False
        f_norm = features / (np.linalg.norm(features) + 1e-8)
        for r in recent[-10:]:  # only check 10 most recent
            r_norm = r / (np.linalg.norm(r) + 1e-8)
            sim = float(np.dot(f_norm, r_norm))
            if sim > 0.98:
                return True
        return False

    def _estimate_importance(self, features: np.ndarray, target: float) -> float:
        """Estimate how informative this sample would be."""
        try:
            pred = self.model.forward(features)
            if isinstance(pred, tuple):
                pred = pred[0]
            err = abs(float(pred) - target)
            return float(min(err, 10.0))  # cap at 10
        except Exception:
            return 1.0  # default importance

    # ── Properties ──

    @property
    def personalization_ready(self) -> bool:
        return self.model.personalization_enabled and self._total_samples >= 500

    @property
    def acceptance_rate(self) -> float:
        total = self._accepted_samples + self._rejected_samples
        return self._accepted_samples / max(total, 1)

    @property
    def mean_loss(self):
        if not self._recent_losses:
            return None
        return float(np.mean(self._recent_losses))

    @property
    def buffer_fill(self) -> float:
        return len(self._buffer) / max(self.buffer_size, 1)
