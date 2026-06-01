"""
DashBox registration — called on device boot.
Replaces sunnylink/utils.py register_sunnylink().
"""

import time

from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from sunnypilot.dashbox import storage

try:
    from dashbox.api import DashboxApi, register_dashbox
except ImportError:
    from openpilot.sunnypilot.dashbox.api import DashboxApi, register_dashbox


NetworkType = None  # Will be imported at runtime


def wait_for_network():
    """Wait until device has network connectivity."""
    from cereal import log, messaging
    NetworkType = log.DeviceState.NetworkType

    rk = Ratekeeper(0.5)
    sm = messaging.SubMaster(['deviceState'], poll='deviceState')
    while True:
        sm.update(1000)
        if sm['deviceState'].networkType != NetworkType.none:
            break
        cloudlog.info(f"DashBox: waiting for network... {time.monotonic()}")
        rk.keep_time()


def main():
    try:
        wait_for_network()
        dongle_id = register_dashbox()
        cloudlog.info(f"DashBox registered: {dongle_id}")
    except Exception:
        cloudlog.exception("DashBox registration failed")
        storage.put("DashboxTempFault", "true")
        raise
