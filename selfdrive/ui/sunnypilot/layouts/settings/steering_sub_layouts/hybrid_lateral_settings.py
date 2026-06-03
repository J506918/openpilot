"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from collections.abc import Callable

import pyray as rl

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp, option_item_sp, LineSeparatorSP
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.network import NavButton
from openpilot.system.ui.widgets.scroller_tici import Scroller


class HybridLateralSettingsLayout(Widget):
  def __init__(self, back_btn_callback: Callable):
    super().__init__()
    self._back_button = NavButton(tr("Back"))
    self._back_button.set_click_callback(back_btn_callback)
    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    self._hybrid_toggle = toggle_item_sp(
      param="HybridLateralControl",
      title=lambda: tr("Hybrid Lateral Control"),
      description=lambda: tr("Enable Hybrid Lateral Control - combines NNLC's predictive lookahead with stock's rock-solid stability."),
    )
    
    self._blend_aggressiveness = option_item_sp(
      param="HybridBlendAggressiveness",
      title=lambda: tr("Blend Aggressiveness"),
      description=lambda: tr("Higher values = more predictive (NNLC-like) behavior on curves. Lower values = more stable (stock-like) behavior."),
      min_value=0,
      max_value=100,
      value_change_step=5,
      label_callback=lambda x: f"{x}%"
    )
    
    self._curve_sensitivity = option_item_sp(
      param="HybridCurveSensitivity",
      title=lambda: tr("Curve Sensitivity"),
      description=lambda: tr("How quickly the controller transitions to predictive mode when detecting curves."),
      min_value=1,
      max_value=100,
      value_change_step=5,
      label_callback=lambda x: f"{x}%"
    )
    
    self._lookahead_distance = option_item_sp(
      param="HybridLookaheadDistance",
      title=lambda: tr("Lookahead Distance"),
      description=lambda: tr("How far ahead (in seconds) to predict curves. Higher = smoother but less responsive."),
      min_value=3,
      max_value=15,
      value_change_step=1,
      label_callback=lambda x: f"{x/10:.1f}s"
    )
    
    self._stability_factor = option_item_sp(
      param="HybridStabilityFactor",
      title=lambda: tr("Straight-Line Stability"),
      description=lambda: tr("Damping factor for straight-line driving. Higher = more stable, lower = more responsive."),
      min_value=50,
      max_value=150,
      value_change_step=5,
      label_callback=lambda x: f"{x/100:.2f}x"
    )

    items = [
      self._hybrid_toggle,
      LineSeparatorSP(40),
      self._blend_aggressiveness,
      self._curve_sensitivity,
      self._lookahead_distance,
      self._stability_factor,
    ]
    return items

  def _update_state(self):
    super()._update_state()
    hybrid_enabled = self._hybrid_toggle.action_item.get_state()
    self._blend_aggressiveness.action_item.set_enabled(ui_state.is_offroad() and hybrid_enabled)
    self._curve_sensitivity.action_item.set_enabled(ui_state.is_offroad() and hybrid_enabled)
    self._lookahead_distance.action_item.set_enabled(ui_state.is_offroad() and hybrid_enabled)
    self._stability_factor.action_item.set_enabled(ui_state.is_offroad() and hybrid_enabled)

  def _render(self, rect):
    self._back_button.set_position(self._rect.x, self._rect.y + 20)
    self._back_button.render()
    # subtract button
    content_rect = rl.Rectangle(rect.x, rect.y + self._back_button.rect.height + 40, rect.width, rect.height - self._back_button.rect.height - 40)
    self._scroller.render(content_rect)

  def show_event(self):
    self._scroller.show_event()
