"""
DashBox API client — registration with DashBox server.
Auth: serial + dongle_id, no token.
"""

import os
import time
import requests

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

DASHBOX_HOST = "https://8.136.28.140:8443"

MAX_REGISTRATION_RETRIES = 10
REGISTRATION_RETRY_DELAY = 10


class DashboxApi:
    """HTTP REST client for DashBox server."""

    def __init__(self):
        self._params = Params()
        self._session = requests.Session()
        self._session.verify = False

    def _url(self, path: str) -> str:
        return f"{DASHBOX_HOST}/{path.lstrip('/')}"

    # ── device registration ───────────────────────────────────────

    def register_device(self) -> str:
        """
        Register device with DashBox server.
        Sends serial + dongle_id; server decides whether to create or return existing.
        Returns dongle_id assigned by server.
        """
        serial = self._read_serial()
        dongle_id = self._params.get("DongleId") or ""

        payload = {
            "serial": serial,
            "dongle_id": dongle_id,
        }

        for attempt in range(1, MAX_REGISTRATION_RETRIES + 1):
            try:
                resp = self._session.post(
                    self._url("api/v1/devices/register"),
                    json=payload,
                    timeout=15,
                )
                if resp.status_code == 200 or resp.status_code == 201:
                    data = resp.json()
                    device_id = data["dongle_id"]
                    self._params.put("DongleId", device_id)
                    cloudlog.info(f"DashBox registered: dongle_id={device_id}")
                    return device_id
                else:
                    cloudlog.warning(
                        f"DashBox registration attempt {attempt}: "
                        f"HTTP {resp.status_code}: {resp.text[:200]}"
                    )
            except Exception as e:
                cloudlog.warning(f"DashBox registration attempt {attempt}: {e}")

            if attempt < MAX_REGISTRATION_RETRIES:
                time.sleep(REGISTRATION_RETRY_DELAY)

        raise Exception(
            f"Failed to register with DashBox after {MAX_REGISTRATION_RETRIES} attempts"
        )

    @staticmethod
    def _read_serial() -> str:
        try:
            with open("/data/params/d/HardwareSerial", "r") as f:
                return f.read().strip()
        except Exception:
            return "unknown"


def register_dashbox() -> str:
    """
    Entry point for device registration.
    Always calls server — server decides if registration is needed.
    """
    api = DashboxApi()
    return api.register_device()
