"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Extension for Hybrid Lateral Control - integrates with controlsd
"""

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog


class HybridLateralControlExt:
  """Extension handler for Hybrid Lateral Control parameters."""
  
  def __init__(self):
    self.params = Params()
    self.frame = -1
    self.hybrid_enabled = self.params.get_bool("HybridLateralControl")
    
    # Default parameter values
    self.blend_aggressiveness = 50  # 0-100
    self.curve_sensitivity = 50      # 0-100
    self.lookahead_distance = 8      # 3-15 (in 0.1s increments)
    self.stability_factor = 100      # 50-150 (in 0.01x increments)
    
    self._load_params()

  def _load_params(self):
    """Load parameters from Params storage."""
    try:
      self.blend_aggressiveness = int(self.params.get("HybridBlendAggressiveness", return_default=True))
    except (ValueError, TypeError):
      cloudlog.warning("HybridLateral: invalid HybridBlendAggressiveness, using default 50")
      self.blend_aggressiveness = 50
    
    try:
      self.curve_sensitivity = int(self.params.get("HybridCurveSensitivity", return_default=True))
    except (ValueError, TypeError):
      cloudlog.warning("HybridLateral: invalid HybridCurveSensitivity, using default 50")
      self.curve_sensitivity = 50
    
    try:
      self.lookahead_distance = int(self.params.get("HybridLookaheadDistance", return_default=True))
    except (ValueError, TypeError):
      cloudlog.warning("HybridLateral: invalid HybridLookaheadDistance, using default 8")
      self.lookahead_distance = 8
    
    try:
      self.stability_factor = int(self.params.get("HybridStabilityFactor", return_default=True))
    except (ValueError, TypeError):
      cloudlog.warning("HybridLateral: invalid HybridStabilityFactor, using default 100")
      self.stability_factor = 100

  def update(self):
    """Update parameters periodically."""
    self.frame += 1
    
    # Reload parameters every 300 frames (10 seconds at 30 Hz)
    if self.frame % 300 == 0:
      self.hybrid_enabled = self.params.get_bool("HybridLateralControl")
      self._load_params()

  def get_blend_factor_scale(self):
    """Get blend factor scaling based on aggressiveness."""
    # 0% = 0.3x (very stable), 50% = 1.0x (balanced), 100% = 1.5x (very predictive)
    return 0.3 + (self.blend_aggressiveness / 100.0) * 1.2

  def get_curve_sensitivity_scale(self):
    """Get curve sensitivity scaling."""
    # 0% = 0.5x (slow transition), 50% = 1.0x (normal), 100% = 1.5x (fast transition)
    return 0.5 + (self.curve_sensitivity / 100.0) * 1.0

  def get_lookahead_time(self):
    """Get lookahead time in seconds."""
    return self.lookahead_distance / 10.0

  def get_stability_factor(self):
    """Get stability factor for straight-line damping."""
    return self.stability_factor / 100.0
