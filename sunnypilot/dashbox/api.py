"""
DashBox API client — replacement for sunnylink/api.py.
Talks to DashBox server at DASHBOX_HOST instead of api.sunnypilot.com.
"""

import json
import os
import time
import requests
from typing import Optional

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from sunnypilot.dashbox import storage

DASHBOX_HOST = "https://8.136.28.140:8443"
UNREGISTERED_DONGLE_ID = "UnregisteredDevice"

# Retry config
MAX_REGISTRATION_RETRIES = 10
REGISTRATION_RETRY_DELAY = 10  # seconds between retries


class DashboxApi:
    """HTTP REST client for DashBox server."""

    def __init__(self, dongle_id: Optional[str] = None):
        self.dongle_id = dongle_id
        self._token: Optional[str] = None
        self._params = Params()
        self._session = requests.Session()
        self._session.verify = False  # self-signed cert on DashBox server

    # ── token management ──────────────────────────────────────────

    def get_token(self) -> Optional[str]:
        if self._token:
            return self._token
        self._token = storage.get("SunnylinkToken")
        return self._token

    def set_token(self, token: str) -> None:
        self._token = token
        storage.put("SunnylinkToken", token)

    # ── HTTP helpers ──────────────────────────────────────────────

    def _url(self, path: str) -> str:
        return f"{DASHBOX_HOST}/{path.lstrip('/')}"

    def _headers(self, access_token: Optional[str] = None) -> dict:
        h = {"Content-Type": "application/json"}
        token = access_token or self.get_token()
        if token:
            h["Authorization"] = f"Bearer {token}"
        return h

    def _get(self, path: str, **kwargs) -> requests.Response:
        return self._session.get(
            self._url(path),
            headers=self._headers(kwargs.pop("access_token", None)),
            timeout=kwargs.pop("timeout", 30),
            **kwargs,
        )

    def _post(self, path: str, data: dict = None, **kwargs) -> requests.Response:
        return self._session.post(
            self._url(path),
            json=data,
            headers=self._headers(kwargs.pop("access_token", None)),
            timeout=kwargs.pop("timeout", 30),
            **kwargs,
        )

    # ── device registration ───────────────────────────────────────

    def register_device(self) -> str:
        """
        Register device with DashBox server.
        Returns the DashBox device_id (DongleId equivalent).
        """
        # Read serial directly from file (avoid params whitelist issues)
        try:
            with open("/data/params/d/HardwareSerial", "r") as f:
                serial = f.read().strip()
        except Exception:
            serial = "unknown"
        imei = "unknown"

        payload = {
            "serial": serial,
            "imei": imei,
            "public_key": "",
        }

        for attempt in range(1, MAX_REGISTRATION_RETRIES + 1):
            try:
                resp = self._session.post(
                    self._url("api/v1/devices/register"),
                    json=payload,
                    timeout=15,
                )
                if resp.status_code == 201:
                    data = resp.json()
                    device_id = data["device_id"]
                    token = data["token"]
                    pairing_code = data.get("pairing_code", "")
                    self.dongle_id = device_id
                    self.set_token(token)
                    Params().put("DongleId", device_id)
                    if pairing_code:
                        storage.put("SunnylinkPairingCode", pairing_code)
                        cloudlog.info(f"DashBox pairing code: {pairing_code}")
                    cloudlog.info(f"DashBox registered: device_id={device_id}")
                    return device_id
                else:
                    cloudlog.warning(
                        f"DashBox registration attempt {attempt}: "
                        f"HTTP {resp.status_code}: {resp.text[:200]}"
                    )
            except Exception as e:
                cloudlog.warning(
                    f"DashBox registration attempt {attempt}: {e}"
                )

            if attempt < MAX_REGISTRATION_RETRIES:
                time.sleep(REGISTRATION_RETRY_DELAY)

        raise Exception(
            f"Failed to register with DashBox after {MAX_REGISTRATION_RETRIES} attempts"
        )

    def _fetch_pairing_code(self, device_id: str) -> None:
        """Fetch a pairing code from DashBox server."""
        try:
            token = self.get_token()
            if not token:
                cloudlog.warning("DashBox: no token for pairing code fetch")
                return
            resp = self._post(
                f"api/v1/devices/{device_id}/pair",
                data={},
                access_token=token,
            )
            if resp.status_code == 200:
                data = resp.json()
                code = data.get("pairing_code", "")
                if code:
                    storage.put("SunnylinkPairingCode", code)
                    cloudlog.info(f"DashBox pairing code: {code}")
            else:
                cloudlog.warning(f"DashBox pairing code fetch failed: {resp.status_code}")
        except Exception as e:
            cloudlog.warning(f"DashBox pairing code fetch error: {e}")

    # ── param sync ────────────────────────────────────────────────

    def get_params(self) -> dict:
        """Fetch all params from server."""
        resp = self._get(f"api/v1/devices/{self.dongle_id}/params")
        if resp.status_code == 200:
            return resp.json().get("params", {})
        cloudlog.warning(f"DashBox get_params failed: {resp.status_code}")
        return {}

    def save_params(self, params: dict) -> bool:
        """Push params to server."""
        resp = self._post(
            f"api/v1/devices/{self.dongle_id}/params",
            data={"params": params},
        )
        ok = resp.status_code in (200, 201)
        if not ok:
            cloudlog.warning(
                f"DashBox save_params failed: {resp.status_code} {resp.text[:200]}"
            )
        return ok

    # ── file upload ───────────────────────────────────────────────

    def get_upload_url(self, path: str) -> dict:
        """Get a pre-signed upload URL for a log file."""
        resp = self._get(
            f"api/v1/devices/{self.dongle_id}/upload_url",
            params={"path": path},
        )
        if resp.status_code == 200:
            return resp.json()
        raise Exception(f"get_upload_url failed: {resp.status_code} {resp.text}")

    # ── health check ──────────────────────────────────────────────

    def health(self) -> bool:
        """Check if server is reachable."""
        try:
            resp = self._session.get(
                self._url("api/v1/health"),
                timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            return False


def register_dashbox() -> str:
    """
    Entry point for device registration.
    Called by dashboxd on boot (like register_sunnylink()).
    """
    existing_id = Params().get("DongleId")
    if existing_id and existing_id != UNREGISTERED_DONGLE_ID:
        token = storage.get("SunnylinkToken")
        if token:
            cloudlog.info(f"DashBox already registered: {existing_id}")
            return existing_id
        # Token missing — re-register to recover
        cloudlog.warning("DashBox: DongleId exists but missing token, re-registering")
        Params().put("DongleId", UNREGISTERED_DONGLE_ID)

    api = DashboxApi()
    device_id = api.register_device()
    # Fetch pairing code after successful registration
    try:
        api._fetch_pairing_code(device_id)
    except Exception:
        cloudlog.exception("DashBox pairing code fetch failed")
    return device_id
