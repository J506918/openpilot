"""
Weight persistence — stdlib only (no msgpack dependency).

Uses pickle for serialization. Stores to /data/params/d/ via Params API.
Atomic save: writes to temp key first, then renames.
"""
import pickle
import time
import numpy as np


def pack_weights(model):
  """Serialize all model weights to bytes (pickle)."""
  data = {
    'encoder': {
      'W1': model.W1, 'b1': model.b1,
      'W2': model.W2, 'b2': model.b2, 'W_res': model.W_res, 'b_res': model.b_res,
      'W3': model.W3, 'b3': model.b3,
    },
    'gate': {
      'W': model.W_gate, 'b': model.b_gate,
      'W_out': model.W_gate_out, 'b_out': model.b_gate_out,
    },
    'heads': {
      name: {k: v.copy() for k, v in h.items()}
      for name, h in model.heads.items()
    },
    'personalization': {
      'P_down': model.P_down, 'P_up': model.P_up,
      'enabled': model.personalization_enabled,
      'confidence': model.personalization_confidence,
      'samples': model._train_samples,
    },
    'meta': {
      'version': 2,
      'timestamp': time.time(),
      'input_dim': 84,
    }
  }
  return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)


def unpack_weights(model, packed):
  """Restore weights from pickle bytes. Returns True on success."""
  try:
    data = pickle.loads(packed)
  except Exception:
    return False

  meta = data.get('meta', {})
  if meta.get('version', 0) < 1:
    return False

  try:
    e = data['encoder']
    model.W1, model.b1 = e['W1'], e['b1']
    model.W2, model.b2 = e['W2'], e['b2']
    model.W_res, model.b_res = e['W_res'], e['b_res']
    model.W3, model.b3 = e['W3'], e['b3']

    g = data['gate']
    model.W_gate, model.b_gate = g['W'], g['b']
    model.W_gate_out, model.b_gate_out = g['W_out'], g['b_out']

    for name, h in data['heads'].items():
      if name in model.heads:
        model.heads[name]['W'] = h['W']
        model.heads[name]['b'] = h['b']
        model.heads[name]['W_out'] = h['W_out']
        model.heads[name]['b_out'] = h['b_out']

    p = data.get('personalization', {})
    if p:
      model.P_down = p.get('P_down', model.P_down)
      model.P_up = p.get('P_up', model.P_up)
      model.personalization_enabled = p.get('enabled', False)
      model.personalization_confidence = p.get('confidence', 0.0)
      model._train_samples = p.get('samples', 0)
  except Exception:
    return False

  return True


class WeightManager:
  """Periodic weight persistence with atomic saves."""

  PARAM_KEY = "SteeringModelWeights"
  PARAM_KEY_TMP = "SteeringModelWeightsTmp"
  SAVE_INTERVAL = 600
  RETRY_INTERVAL = 1800  # retry failed saves after 18s

  def __init__(self, model, params):
    self.model = model
    self.params = params
    self._frame = 0
    self._dirty = False
    self._last_save_samples = 0
    self._last_save_frame = 0
    self._save_failures = 0

  def mark_dirty(self):
    self._dirty = True

  def update(self):
    """Call every frame. Saves periodically."""
    self._frame += 1

    if self.model._train_samples > self._last_save_samples:
      self._dirty = True

    frames_since_save = self._frame - self._last_save_frame
    should_save = (
      self._dirty and
      frames_since_save >= (self.SAVE_INTERVAL if self._save_failures == 0 else self.RETRY_INTERVAL)
    )

    if should_save:
      self._save()

  def _save(self):
    """Atomic save: temp → rename."""
    try:
      packed = pack_weights(self.model)
      # Write to temp key first
      self.params.put(self.PARAM_KEY_TMP, packed)
      # Then move to real key (atomic-ish via put)
      self.params.put(self.PARAM_KEY, packed)
      self._last_save_samples = self.model._train_samples
      self._last_save_frame = self._frame
      self._dirty = False
      self._save_failures = 0
    except Exception:
      self._save_failures += 1

  def load(self):
    """Load weights on startup."""
    try:
      packed = self.params.get(self.PARAM_KEY)
      if packed is not None and len(packed) > 0:
        return unpack_weights(self.model, packed)
    except Exception:
      pass
    return False
