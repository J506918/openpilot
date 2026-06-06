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


class HybridLateralSettingsV2Layout(Widget):
  def __init__(self, back_btn_callback: Callable):
    super().__init__()
    self._back_button = NavButton(tr("Back"))
    self._back_button.set_click_callback(back_btn_callback)
    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    self._hybrid_v2_toggle = toggle_item_sp(
      param="HybridLateralControlV2",
      title=lambda: tr("Hybrid Lateral Control V2"),
      description=lambda: tr(
        "Model-driven lateral control. Direct torque mapping from model predictions "
        "with slow bias correction. No PID — model draws the line, controller follows."
      ),
    )

    self._lookahead_distance = option_item_sp(
      param="HybridLookaheadDistance",
      title=lambda: tr("Lookahead Distance"),
      description=lambda: tr(
        "How far ahead (in seconds) the model looks to predict curvature. "
        "Higher = smoother but less responsive to sudden curves."
      ),
      min_value=3,
      max_value=15,
      value_change_step=1,
      label_callback=lambda x: f"{x/10:.1f}s"
    )

    items = [
      self._hybrid_v2_toggle,
      LineSeparatorSP(40),
      self._lookahead_distance,
    ]
    return items

  def _update_state(self):
    super()._update_state()
    hybrid_v2_enabled = self._hybrid_v2_toggle.action_item.get_state()
    self._lookahead_distance.action_item.set_enabled(ui_state.is_offroad() and hybrid_v2_enabled)

  def _render(self, rect):
    self._back_button.set_position(self._rect.x, self._rect.y + 20)
    self._back_button.render()
    content_rect = rl.Rectangle(
      rect.x, rect.y + self._back_button.rect.height + 40,
      rect.width, rect.height - self._back_button.rect.height - 40
    )
    self._scroller.render(content_rect)

  def show_event(self):
    self._scroller.show_event()
