"""
DashBox pairing dialog for TICI — displays a pairing code for the DashBox mobile app.
"""
import pyray as rl

from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.widgets.pairing_dialog import PairingDialog
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.lib.text_measure import measure_text_cached
from sunnypilot.dashbox import storage


class DashboxPairingDialog(PairingDialog):
  """DashBox pairing dialog — shows pairing code for the mobile app."""

  def __init__(self):
    PairingDialog.__init__(self)
    self.params = Params()
    self._is_paired_prev = ui_state.sunnylink_state.is_paired()

  @staticmethod
  def _get_pairing_code() -> str:
    return storage.get("SunnylinkPairingCode").strip()

  def _update_state(self):
    is_paired = ui_state.sunnylink_state.is_paired()
    if not self._is_paired_prev and is_paired:
      gui_app.pop_widget()
    self._is_paired_prev = is_paired

  def _render(self, rect: rl.Rectangle) -> int:
    rl.clear_background(rl.Color(224, 224, 224, 255))

    margin = 70
    content_rect = rl.Rectangle(rect.x + margin, rect.y + margin, rect.width - 2 * margin, rect.height - 2 * margin)
    y = content_rect.y

    # Close button
    close_size = 80
    pad = 20
    close_rect = rl.Rectangle(content_rect.x - pad, y - pad, close_size + pad * 2, close_size + pad * 2)
    self._close_btn.render(close_rect)
    y += close_size + 40

    # Title
    title = tr("Pair your device with DashBox")
    title_font = gui_app.font(FontWeight.NORMAL)
    left_width = int(content_rect.width * 0.5 - 15)

    title_wrapped = wrap_text(title_font, title, 75, left_width)
    rl.draw_text_ex(title_font, "\n".join(title_wrapped), rl.Vector2(content_rect.x, y), 75, 0.0, rl.BLACK)
    y += len(title_wrapped) * 75 + 60

    remaining_height = content_rect.height - (y - content_rect.y)
    right_width = content_rect.width // 2 - 20

    # Instructions (left column)
    self._render_instructions(rl.Rectangle(content_rect.x, y, left_width, remaining_height))

    # Pairing code display (right column, centered)
    code = self._get_pairing_code()
    code_x = content_rect.x + left_width + 40
    code_area_width = right_width
    code_center_x = code_x + code_area_width // 2

    if code:
      # Large pairing code
      code_font = gui_app.font(FontWeight.DISPLAY)
      code_size = measure_text_cached(code_font, code, 80)
      rl.draw_text_ex(
        code_font, code,
        rl.Vector2(code_center_x - code_size.x // 2, content_rect.y + content_rect.height // 2 - 60),
        80, 0.0, rl.Color(0, 180, 0, 255),
      )

      # Subtitle
      sub_font = gui_app.font(FontWeight.ROMAN)
      sub_text = tr("Pairing Code")
      sub_size = measure_text_cached(sub_font, sub_text, 36)
      rl.draw_text_ex(
        sub_font, sub_text,
        rl.Vector2(code_center_x - sub_size.x // 2, content_rect.y + content_rect.height // 2 + 40),
        36, 0.0, rl.Color(128, 128, 128, 255),
      )
    else:
      loading_font = gui_app.font(FontWeight.NORMAL)
      loading_text = tr("Loading...")
      ld_size = measure_text_cached(loading_font, loading_text, 48)
      rl.draw_text_ex(
        loading_font, loading_text,
        rl.Vector2(code_center_x - ld_size.x // 2, content_rect.y + content_rect.height // 2),
        48, 0.0, rl.Color(128, 128, 128, 255),
      )

    return -1

  def _render_instructions(self, rect: rl.Rectangle) -> None:
    instructions = [
      tr("Open the DashBox app on your phone"),
      tr("Tap \"Add Device\" and enter the pairing code"),
      tr("The code is shown on the right"),
      tr("The device will update its status once paired"),
    ]

    font = gui_app.font(FontWeight.BOLD)
    y = rect.y

    for i, text in enumerate(instructions):
      circle_radius = 25
      circle_x = rect.x + circle_radius + 15
      text_x = rect.x + circle_radius * 2 + 40
      text_width = rect.width - (circle_radius * 2 + 40)

      wrapped = wrap_text(font, text, 47, int(text_width))
      text_height = len(wrapped) * 47
      circle_y = y + text_height // 2

      rl.draw_circle(int(circle_x), int(circle_y), circle_radius, rl.Color(70, 70, 70, 255))
      number = str(i + 1)
      number_size = measure_text_cached(font, number, 30)
      rl.draw_text_ex(font, number, (int(circle_x - number_size.x // 2), int(circle_y - number_size.y // 2)), 30, 0, rl.WHITE)

      rl.draw_text_ex(font, "\n".join(wrapped), rl.Vector2(text_x, y), 47, 0.0, rl.BLACK)
      y += text_height + 50


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
