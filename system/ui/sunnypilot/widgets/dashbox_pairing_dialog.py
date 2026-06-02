"""
DashBox pairing dialog for TICI — calls server directly, no local caching.
"""
import json
import threading
import pyray as rl
import urllib.request
import ssl

from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.widgets.pairing_dialog import PairingDialog
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.lib.text_measure import measure_text_cached

DASHBOX_URL = "https://8.136.28.140:8443/api/v1/devices/pair"


class DashboxPairingDialog(PairingDialog):
  """DashBox pairing dialog — requests pairing code from server via HTTP."""

  def __init__(self):
    PairingDialog.__init__(self)
    self.params = Params()
    self._is_paired_prev = ui_state.sunnylink_state.is_paired()
    self._code = ""
    self._error = ""

    # Request pairing code in background — no storage, no dashboxd
    threading.Thread(target=self._request_code, daemon=True).start()

  def _request_code(self):
    try:
      serial = self._read_serial()
      dongle_id = self._read_dongle_id()
      body = json.dumps({"serial": serial, "dongle_id": dongle_id}).encode()

      ctx = ssl.create_default_context()
      ctx.check_hostname = False
      ctx.verify_mode = ssl.CERT_NONE

      req = urllib.request.Request(DASHBOX_URL, data=body, headers={"Content-Type": "application/json"})
      resp = urllib.request.urlopen(req, context=ctx, timeout=10)
      data = json.loads(resp.read())
      self._code = data.get("pairing_code", "")
    except Exception as e:
      self._error = str(e)

  @staticmethod
  def _read_serial() -> str:
    try:
      with open("/data/params/d/HardwareSerial", "r") as f:
        return f.read().strip()
    except Exception:
      return ""

  @staticmethod
  def _read_dongle_id() -> str:
    try:
      from sunnypilot.dashbox import storage
      return storage.get("DongleId").strip()
    except Exception:
      return ""

  def _get_pairing_code(self) -> str:
    if self._error:
      return ""
    return self._code

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
