from openpilot.common.params import Params
"""
DashBox settings page — standalone layout with own branding and flow.
"""
import time
import pyray as rl
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import button_item_sp, toggle_item_sp
from openpilot.system.ui.sunnypilot.widgets.dashbox_pairing_dialog import DashboxPairingDialog
from sunnypilot.dashbox import storage
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.system.ui.widgets.scroller_tici import Scroller, LineSeparator

UNREGISTERED_DONGLE_ID = "UnregisteredDevice"


class DashBoxHeader(Widget):
  """DashBox branding header."""
  def __init__(self):
    super().__init__()
    self._title = UnifiedLabel(
      text="DashBox",
      font_size=80,
      font_weight=FontWeight.AUDIOWIDE,
      text_color=rl.WHITE,
      alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER,
      alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_TOP,
    )
    self._subtitle = UnifiedLabel(
      text=tr("Device management & remote access"),
      font_size=36,
      font_weight=FontWeight.NORMAL,
      text_color=rl.Color(0, 200, 200, 255),
      alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER,
      alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_TOP,
      wrap_text=True,
    )
    self._padding = 20
    self._spacing = 12

  def set_parent_rect(self, parent_rect: rl.Rectangle) -> None:
    super().set_parent_rect(parent_rect)
    cw = int(parent_rect.width - self._padding * 2)
    th = self._title.get_content_height(cw)
    sh = self._subtitle.get_content_height(cw)
    self._rect.width = parent_rect.width
    self._rect.height = self._padding + th + self._spacing + sh + self._padding

  def _render(self, rect: rl.Rectangle):
    cw = rect.width - self._padding * 2
    cy = rect.y + self._padding
    th = self._title.get_content_height(int(cw))
    self._title.render(rl.Rectangle(rect.x + self._padding, cy, cw, th))
    cy += th + self._spacing
    sh = self._subtitle.get_content_height(int(cw))
    self._subtitle.render(rl.Rectangle(rect.x + self._padding, cy, cw, sh))


class DashBoxLayout(Widget):
  """DashBox settings — clean, own branding, own flow."""

  def __init__(self):
    super().__init__()
    self._pairing_dialog: DashboxPairingDialog | None = None
    self._enabled = ui_state.params.get_bool("SunnylinkEnabled")

    items = self._build_items()
    self._scroller = Scroller(items, line_separator=False, spacing=0)

  def _build_items(self):
    self._toggle = toggle_item_sp(
      title=tr("Enable DashBox"),
      description=tr("Master switch for DashBox remote access and management."),
      param="SunnylinkEnabled",
      callback=self._on_toggle,
    )

    self._device_id_btn = button_item_sp(
      title=tr("Device ID"),
      button_text=tr("N/A"),
      description=tr("Unique identifier for this device."),
      callback=None,
    )
    self._device_id_btn.action_item.set_enabled(False)

    self._pair_btn = button_item_sp(
      title=tr("Pair Device"),
      button_text=tr("Pair"),
      description=tr("Generate QR code to pair with DashBox mobile app."),
      callback=self._on_pair,
    )

    self._status_label = button_item_sp(
      title=tr("Connection"),
      button_text=tr("OFFLINE"),
      description=tr("DashBox server connection status."),
      callback=None,
    )
    self._status_label.action_item.set_enabled(False)

    return [
      DashBoxHeader(),
      LineSeparator(),
      self._toggle,
      LineSeparator(),
      self._device_id_btn,
      LineSeparator(),
      self._pair_btn,
      LineSeparator(),
      self._status_label,
    ]

  @staticmethod
  def _dongle_id() -> str:
    did = Params().get("DongleId")
    if not did or did == UNREGISTERED_DONGLE_ID:
      return tr("N/A")
    if len(did) > 16:
      return did[:8] + "..." + did[-8:]
    return did

  def _on_toggle(self, state: bool):
    self._enabled = state
    self._refresh()

  def _on_pair(self):
    did = self._dongle_id()
    if did == tr("N/A") or did == UNREGISTERED_DONGLE_ID:
      return  # no device ID yet
    # MICI: check is_dismissing; TICI: AttributeError → always allow
    try:
      if self._pairing_dialog and not self._pairing_dialog.is_dismissing:
        return
    except AttributeError:
      pass
    # Ask server for pairing code (server reuses valid code, generates if expired)
    try:
      from sunnypilot.dashbox.api import DashboxApi
      dongle_id = Params().get("DongleId")
      if dongle_id:
        api = DashboxApi(dongle_id)
        api._fetch_pairing_code(dongle_id)
    except Exception:
      pass
    self._pairing_dialog = DashboxPairingDialog()
    gui_app.push_widget(self._pairing_dialog)

  def _refresh(self):
    self._device_id_btn.action_item.set_text(self._dongle_id())
    can_act = self._enabled and not ui_state.is_onroad()
    self._pair_btn.action_item.set_enabled(can_act)
    self._toggle.action_item.set_enabled(not ui_state.is_onroad())

    # Connection status from heartbeat
    if self._enabled:
      try:
        with open("/data/params/d/DashboxOnline", "r") as f:
          is_online = f.read().strip() == "1"
      except Exception:
        is_online = False
      self._status_label.action_item.set_text(tr("ONLINE") if is_online else tr("OFFLINE"))
    else:
      self._status_label.action_item.set_text(tr("DISABLED"))

  def _update_state(self):
    super()._update_state()
    self._enabled = ui_state.params.get_bool("SunnylinkEnabled")
    self._toggle.action_item.set_state(self._enabled)
    self._refresh()

  def _render(self, rect):
    self._scroller.render(rect)

  def show_event(self):
    super().show_event()
    self._scroller.show_event()

  def hide_event(self):
    super().hide_event()
