"""
DashBox state management — replaces sunnylink_state.py.
No membership/roles — all features free.
"""

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from sunnypilot.dashbox import storage

try:
    from dashbox.api import DashboxApi
except ImportError:
    from openpilot.sunnypilot.dashbox.api import DashboxApi


class DashboxState:
    def __init__(self):
        self._params = Params()
        self.dongle_id = Params().get("DongleId")
        self._api = DashboxApi(self.dongle_id)
        self.connected = False

    def check_connection(self) -> bool:
        """Check if DashBox server is reachable."""
        self.connected = self._api.health()
        return self.connected

    def sync_params(self, local_params: dict) -> bool:
        """Push local params to server."""
        if not self.dongle_id:
            return False
        try:
            return self._api.save_params(local_params)
        except Exception:
            cloudlog.exception("DashBox sync_params failed")
            return False

    def pull_params(self) -> dict:
        """Pull params from server to device."""
        if not self.dongle_id:
            return {}
        try:
            return self._api.get_params()
        except Exception:
            cloudlog.exception("DashBox pull_params failed")
            return {}
