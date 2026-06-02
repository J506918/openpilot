"""
DashBox pairing dialog for MICI — displays a pairing code for the DashBox mobile app.
"""
import pyray as rl

from openpilot.common.params import Params
from openpilot.selfdrive.ui.mici.widgets.pairing_dialog import PairingDialog
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets.nav_widget import NavWidget
from openpilot.system.ui.widgets.label import UnifiedLabel
from sunnypilot.dashbox import storage


class DashboxPairingDialog(PairingDialog):
  """DashBox pairing dialog — shows pairing code for the mobile app."""

  def __init__(self):
    PairingDialog.__init__(self)
    self._params = Params()
    self._is_paired_prev = False
    self._requested = False

    # Clear old code and signal dashboxd to request a fresh one from server
    storage.put("SunnylinkPairingCode", "")
    self._params.put("DashboxRequestPairing", "1")

    self._title_label = UnifiedLabel(
      tr("Pair with DashBox"),
      font_size=52,
      font_weight=FontWeight.BOLD,
      text_color=rl.Color(255, 255, 255, int(255 * 0.9)),
      line_height=0.8,
    )

    self._instruction_label = UnifiedLabel(
      tr("Enter this code in the DashBox app:"),
      font_size=34,
      font_weight=FontWeight.NORMAL,
      text_color=rl.Color(255, 255, 255, int(255 * 0.6)),
      line_height=0.8,
    )

    code = self._get_pairing_code()
    self._code_label = UnifiedLabel(
      code or tr("Loading..."),
      font_size=64,
      font_weight=FontWeight.DISPLAY,
      text_color=rl.Color(0, 220, 0, 255),
      line_height=0.8,
    )

  @staticmethod
  def _get_pairing_code() -> str:
    return storage.get("SunnylinkPairingCode").strip()

  def _update_state(self):
    NavWidget._update_state(self)

    is_paired = ui_state.sunnylink_state.is_paired()
    if not self._is_paired_prev and is_paired and not self.is_dismissing:
      self.dismiss()
    self._is_paired_prev = is_paired

    code = self._get_pairing_code()
    if code:
      self._code_label.set_text(code)

  def _render(self, rect: rl.Rectangle):
    center_x = self._rect.x + self._rect.width // 2

    # Title
    self._title_label.set_max_width(int(self._rect.width - 40))
    self._title_label.set_position(center_x - self._rect.width // 2 + 20, self._rect.y + 40)
    self._title_label.render()

    # Instruction
    self._instruction_label.set_max_width(int(self._rect.width - 40))
    self._instruction_label.set_position(center_x - self._rect.width // 2 + 20, self._rect.y + 110)
    self._instruction_label.render()

    # Pairing code — large, centered
    code = self._get_pairing_code()
    if code:
      self._code_label.set_max_width(int(self._rect.width - 40))
      self._code_label.set_position(center_x - 120, self._rect.y + 180)
      self._code_label.render()

      # Subtitle
      subtitle_font = gui_app.font(FontWeight.ROMAN)
      subtitle = tr("Pairing Code")
      rl.draw_text_ex(
        subtitle_font, subtitle,
        rl.Vector2(center_x - 60, self._rect.y + self._rect.height - 36),
        28, 0.0,
        rl.Color(255, 255, 255, int(255 * 0.35)),
      )


if __name__ == "__main__":
  gui_app.init_window("pairing device")
  pairing = DashboxPairingDialog()
  try:
    for _ in gui_app.render():
      result = pairing.render(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
      if result != -1:
        break
  finally:
    del pairing
