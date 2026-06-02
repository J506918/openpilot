"""
DashBox pairing dialog for MICI — calls server directly, no local caching.
"""
import json
import threading
import pyray as rl
import urllib.request
import ssl

from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets.nav_widget import NavWidget
from openpilot.system.ui.widgets.label import UnifiedLabel

DASHBOX_URL = "https://8.136.28.140:8443/api/v1/devices/pair"


class DashboxPairingDialog(NavWidget):
  """DashBox pairing dialog — requests pairing code from server via HTTP."""

  def __init__(self):
    NavWidget.__init__(self)
    self._params = Params()
    self._is_paired_prev = False
    self._code = ""
    self._error = ""

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

    self._code_label = UnifiedLabel(
      tr("Loading..."),
      font_size=64,
      font_weight=FontWeight.DISPLAY,
      text_color=rl.Color(0, 220, 0, 255),
      line_height=0.8,
    )

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

  def _update_state(self):
    NavWidget._update_state(self)

    is_paired = ui_state.sunnylink_state.is_paired()
    if not self._is_paired_prev and is_paired and not self.is_dismissing:
      self.dismiss()
    self._is_paired_prev = is_paired

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

    # Pairing code or error
    if self._code:
      self._code_label.set_text(self._code)
    elif self._error:
      self._code_label.set_text(tr("Error"))
      self._code_label.set_text_color(rl.Color(255, 80, 80, 255))
    else:
      self._code_label.set_text(tr("Loading..."))

    self._code_label.set_max_width(int(self._rect.width - 40))
    self._code_label.set_position(center_x - 120, self._rect.y + 180)
    self._code_label.render()

    # Subtitle
    subtitle_font = gui_app.font(FontWeight.ROMAN)
    subtitle = tr("Pairing Code") if self._code else ""
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
