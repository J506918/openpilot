from openpilot.common.params import Params
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import pyray as rl

from collections.abc import Callable

from cereal import custom
from openpilot.selfdrive.ui.mici.widgets.button import BigButton, BigToggle
from openpilot.selfdrive.ui.mici.widgets.dialog import BigDialog, BigConfirmationDialog
from openpilot.selfdrive.ui.sunnypilot.mici.layouts.onboarding import SunnylinkConsentPage
from openpilot.selfdrive.ui.sunnypilot.mici.widgets.dashbox_pairing_dialog import DashboxPairingDialog
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.sunnypilot.sunnylink.api import UNREGISTERED_SUNNYLINK_DONGLE_ID
from openpilot.system.ui.lib.application import gui_app, MousePos, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.system.ui.widgets.scroller import NavScroller
from openpilot.system.version import sunnylink_consent_version, sunnylink_consent_declined

class SunnylinkInfo(Widget):
  def __init__(self):
    super().__init__()

    self.set_rect(rl.Rectangle(0, 0, 360, 180))

    header_color = rl.Color(255, 255, 255, int(255 * 0.9))
    subheader_color = rl.Color(255, 255, 255, int(255 * 0.9 * 0.65))
    max_width = int(self._rect.width - 20)
    self.device_id_header = UnifiedLabel(tr("device id"), 48, max_width=max_width, text_color=header_color,
                                         font_weight=FontWeight.DISPLAY, shimmer=True)
    self.device_id_text = UnifiedLabel(UNREGISTERED_SUNNYLINK_DONGLE_ID, 28, max_width=max_width, text_color=subheader_color,
                                       font_weight=FontWeight.ROMAN, wrap_text=True)

    self.sponsor_header = UnifiedLabel(tr("sponsor tier"), 48, max_width=max_width, text_color=header_color,
                                       font_weight=FontWeight.DISPLAY, shimmer=True)
    self.sponsor_text = UnifiedLabel("N/A", 32, max_width=max_width, text_color=subheader_color, font_weight=FontWeight.ROMAN)

  def _render(self, _):
    self.device_id_header.set_position(self._rect.x + 20, self._rect.y - 10)
    self.device_id_header.render()

    self.device_id_text.set_position(self._rect.x + 20, self._rect.y + 62 - 25)
    self.device_id_text.render()

    self.sponsor_header.set_position(self._rect.x + 20, self._rect.y + 118 - 30)
    self.sponsor_header.render()

    self.sponsor_text.set_position(self._rect.x + 20, self._rect.y + 165 - 25)
    self.sponsor_text.render()

class SunnylinkLayoutMici(NavScroller):
  def __init__(self, back_callback: Callable):
    super().__init__()
    self.set_back_callback(back_callback)
    self._restore_in_progress = False
    self._backup_in_progress = False
    self._dashbox_enabled = ui_state.params.get("SunnylinkEnabled")

    self._dashbox_info = SunnylinkInfo()

    self._dashbox_toggle = BigToggle(text=tr("enable DashBox"),
                                     initial_state=self._dashbox_enabled,
                                     toggle_callback=self._dashbox_toggle_callback)
    self._dashbox_sponsor_button = SunnylinkPairBigButton(sponsor_pairing=False)
    self._dashbox_pair_button = SunnylinkPairBigButton(sponsor_pairing=True)
    self._backup_btn = BigButton(tr("backup settings"), "")
    self._backup_btn.set_click_callback(lambda: self._handle_backup_restore_btn(restore=False))
    self._restore_btn = BigButton(tr("restore settings"), "")
    self._restore_btn.set_click_callback(lambda: self._handle_backup_restore_btn(restore=True))
    self._dashbox_uploader_toggle = BigToggle(text=tr("DashBox uploader"), initial_state=False,
                                              toggle_callback=self._dashbox_uploader_callback)

    self._scroller.add_widgets([
      self._dashbox_info,
      self._dashbox_toggle,
      self._dashbox_sponsor_button,
      self._dashbox_pair_button,
      self._backup_btn,
      self._restore_btn,
      self._dashbox_uploader_toggle
    ])

  def _update_state(self):
    super()._update_state()
    self._dashbox_enabled = ui_state.params.get("SunnylinkEnabled")
    self._dashbox_toggle.set_checked(self._dashbox_enabled)
    self._dashbox_pair_button.set_visible(self._dashbox_enabled)
    self._dashbox_sponsor_button.set_visible(self._dashbox_enabled)
    self._backup_btn.set_visible(self._dashbox_enabled)
    self._restore_btn.set_visible(self._dashbox_enabled)
    self._dashbox_uploader_toggle.set_visible(self._dashbox_enabled)
    self.handle_backup_restore_progress()

    # Format device ID with newline at midpoint for display
    _raw_id = Params().get("DongleId") or UNREGISTERED_SUNNYLINK_DONGLE_ID
    _mid = len(_raw_id) // 2
    _formatted_id = _raw_id[:_mid] + "\n" + _raw_id[_mid:]
    self._dashbox_info.device_id_text.set_text(_formatted_id)
    self._dashbox_info.sponsor_text.set_text(ui_state.sunnylink_state.get_sponsor_tier().name.lower() or "N/A")
    self._dashbox_info.set_visible(self._dashbox_enabled)

    if ui_state.sunnylink_state.is_sponsor():
      self._dashbox_sponsor_button.set_text(tr("thanks"))
      self._dashbox_sponsor_button.set_value(ui_state.sunnylink_state.get_sponsor_tier().name.lower())
      self._dashbox_sponsor_button.set_enabled(False)
    else:
      self._dashbox_sponsor_button.set_text(tr("sponsor"))
      self._dashbox_sponsor_button.set_value("")

    if ui_state.sunnylink_state.is_paired():
      self._dashbox_pair_button.set_text(tr("paired"))
    else:
      self._dashbox_pair_button.set_text(tr("pair"))

    state_map = {"connecting": "Connecting…", "registering": "Registering…", "online": "Online", "offline": "Offline"}
    state_text = tr(state_map.get(ui_state.sunnylink_state.dashbox_state, "Offline"))
    self._dashbox_pair_button.set_value(state_text)

  def show_event(self):
    super().show_event()
    ui_state.update_params()
    ui_state.sunnylink_state.set_settings_open(True)

  def hide_event(self):
    super().hide_event()
    ui_state.sunnylink_state.set_settings_open(False)

  @staticmethod
  def _dashbox_toggle_callback(state: bool):
    sl_consent: bool = ui_state.params.get("CompletedSunnylinkConsentVersion") == sunnylink_consent_version
    sl_enabled: bool = ui_state.params.get("SunnylinkEnabled")

    def sl_terms_accepted():
      ui_state.params.put("CompletedSunnylinkConsentVersion", sunnylink_consent_version)
      ui_state.params.put_bool("SunnylinkEnabled", True)
      gui_app.pop_widget()

    def sl_terms_declined():
      ui_state.params.put("CompletedSunnylinkConsentVersion", sunnylink_consent_declined)
      ui_state.params.put_bool("SunnylinkEnabled", False)
      gui_app.pop_widget()

    if state and not sl_consent and not sl_enabled:
      sl_terms_dlg = SunnylinkConsentPage(on_accept=sl_terms_accepted, on_decline=sl_terms_declined)
      gui_app.push_widget(sl_terms_dlg)
    else:
      ui_state.params.put_bool("SunnylinkEnabled", state)

    ui_state.update_params()

  @staticmethod
  def _dashbox_uploader_callback(state: bool):
    ui_state.params.put_bool("EnableSunnylinkUploader", state)

  def _handle_backup_restore_btn(self, restore: bool = False):
    lbl = tr("slide to restore") if restore else tr("slide to backup")
    icon = gui_app.texture("icons_mici/settings/device/update.png", 64, 64)
    dlg = BigConfirmationDialog(lbl, icon, confirm_callback=self._restore_handler if restore else self._backup_handler)
    gui_app.push_widget(dlg)

  def _backup_handler(self):
    self._backup_in_progress = True
    self._backup_btn.set_enabled(False)
    ui_state.params.put_bool("BackupManager_CreateBackup", True)

  def _restore_handler(self):
    self._restore_in_progress = True
    self._restore_btn.set_enabled(False)
    ui_state.params.put("BackupManager_RestoreVersion", "latest")

  def handle_backup_restore_progress(self):
    dashbox_backup_manager = ui_state.sm["backupManagerSP"]

    backup_status = dashbox_backup_manager.backupStatus
    restore_status = dashbox_backup_manager.restoreStatus
    backup_progress = dashbox_backup_manager.backupProgress
    restore_progress = dashbox_backup_manager.restoreProgress

    if self._backup_in_progress:
      self._restore_btn.set_enabled(False)
      self._backup_btn.set_enabled(False)

      if backup_status == custom.BackupManagerSP.Status.inProgress:
        self._backup_in_progress = True
        self._backup_btn.set_text(tr("backing up"))
        text = tr(f"{backup_progress}%")
        self._backup_btn.set_value(text)

      elif backup_status == custom.BackupManagerSP.Status.failed:
        self._backup_in_progress = False
        self._backup_btn.set_enabled(not ui_state.is_onroad())
        self._backup_btn.set_text(tr("backup"))
        self._backup_btn.set_value(tr("failed"))

      elif (backup_status == custom.BackupManagerSP.Status.completed or
            (backup_status == custom.BackupManagerSP.Status.idle and backup_progress == 100.0)):
        self._backup_in_progress = False
        gui_app.push_widget(BigDialog(title=tr("settings backed up"), description=""))
        self._backup_btn.set_enabled(not ui_state.is_onroad())

    elif self._restore_in_progress:
      self._restore_btn.set_enabled(False)
      self._backup_btn.set_enabled(False)

      if restore_status == custom.BackupManagerSP.Status.inProgress:
        self._restore_in_progress = True
        self._restore_btn.set_text(tr("restoring"))
        text = tr(f"{restore_progress}%")
        self._restore_btn.set_value(text)

      elif restore_status == custom.BackupManagerSP.Status.failed:
        self._restore_in_progress = False
        self._restore_btn.set_enabled(not ui_state.is_onroad())
        self._restore_btn.set_text(tr("restore"))
        self._restore_btn.set_value(tr("failed"))
        gui_app.push_widget(BigDialog(title=tr("unable to restore"), description="try again later."))

      elif (restore_status == custom.BackupManagerSP.Status.completed or
            (restore_status == custom.BackupManagerSP.Status.idle and restore_progress == 100.0)):
        self._restore_in_progress = False
        gui_app.push_widget(BigConfirmationDialog(
          title="slide to restart", icon=gui_app.texture("icons_mici/settings/device/reboot.png", 64, 64),
          confirm_callback=lambda: gui_app.request_close()))

    else:
      can_enable = self._dashbox_enabled and not ui_state.is_onroad()
      self._backup_btn.set_enabled(can_enable)
      self._backup_btn.set_text(tr("backup settings"))
      self._backup_btn.set_value("")
      self._restore_btn.set_enabled(can_enable)
      self._restore_btn.set_text(tr("restore settings"))
      self._restore_btn.set_value("")


class SunnylinkPairBigButton(BigButton):
  def __init__(self, sponsor_pairing: bool = False):
    self.sponsor_pairing = sponsor_pairing
    super().__init__("")

  def _update_state(self):
    super()._update_state()

  def _handle_mouse_release(self, mouse_pos: MousePos):
    super()._handle_mouse_release(mouse_pos)

    network_type = ui_state.sm["deviceState"].networkType

    dlg: BigDialog | DashboxPairingDialog | None = None

    if network_type == 0:
      dlg = BigDialog(tr("no internet"), tr("please connect to WiFi & try again"))
    elif UNREGISTERED_SUNNYLINK_DONGLE_ID == (Params().get("DongleId") or UNREGISTERED_SUNNYLINK_DONGLE_ID):
      dlg = BigDialog(tr("DashBox device id not found"), tr("please reboot & try again"))
    elif self.sponsor_pairing:
      dlg = DashboxPairingDialog()
    elif not self.sponsor_pairing:
      dlg = DashboxPairingDialog()
    if dlg:
      gui_app.push_widget(dlg)
