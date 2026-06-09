import json
import os
import random
import time
import jwt
from typing import cast
from datetime import datetime, timedelta, UTC

from openpilot.common.api.base import BaseApi
from openpilot.common.params import Params
from openpilot.system.hardware import HARDWARE
from openpilot.system.hardware.hw import Paths

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_HOST = os.getenv('SUNNYLINK_API_HOST', 'https://8.136.28.140:8443/api/v1')
UNREGISTERED_SUNNYLINK_DONGLE_ID = "UnregisteredDevice"
MAX_RETRIES = 6
CRASH_LOG_DIR = Paths.crash_log_root()


class SunnylinkApi(BaseApi):
  def __init__(self, dongle_id):
    super().__init__(dongle_id, API_HOST)
    self.user_agent = "sunnypilot-"
    self.spinner = None
    self.params = Params()

  def api_get(self, endpoint, method='GET', timeout=10, access_token=None, session=None, json=None, **kwargs):
    if not self.params.get_bool("SunnylinkEnabled"):
      return None

    return super().api_get(endpoint, method, timeout, access_token, session, json, **kwargs)

  def get_token(self, payload_extra=None, expiry_hours=1):
    # Add your additional data here
    additional_data = {}
    return super()._get_token(payload_extra, expiry_hours, **additional_data)

  def _status_update(self, message):
    print(message)
    if self.spinner:
      self.spinner.update(message)
      time.sleep(0.5)

  def _resolve_dongle_ids(self):
    sunnylink_dongle_id = self.params.get("DongleId")
    comma_dongle_id = self.dongle_id or self.params.get("DongleId")
    return sunnylink_dongle_id, comma_dongle_id

  def _resolve_imeis(self):
    imei1, imei2 = None, None
    imei_try = 0
    while imei1 is None and imei2 is None and imei_try < MAX_RETRIES:
      try:
        imei1, imei2 = HARDWARE.get_imei(0), HARDWARE.get_imei(1)
      except Exception:
        self._status_update(f"Error getting imei, trying again... [{imei_try + 1}/{MAX_RETRIES}]")
        time.sleep(1)
      imei_try += 1
    return imei1, imei2

  def _resolve_serial(self):
    return (self.params.get("HardwareSerial")
            or HARDWARE.get_serial())

  # ── Direct DashBox HTTP helpers ─────────────────────────────────

  def _dash_post(self, path, data, token=None):
    h = {"Content-Type": "application/json"}
    if token:
      h["Authorization"] = f"Bearer {token}"
    return requests.post(f"{API_HOST}/{path}", json=data, headers=h,
                         timeout=15, verify=False)

  def _dash_get(self, path, token=None):
    h = {}
    if token:
      h["Authorization"] = f"Bearer {token}"
    return requests.get(f"{API_HOST}/{path}", headers=h,
                        timeout=10, verify=False)

  # ── Registration ───────────────────────────────────────────────

  def register_device(self, spinner=None, timeout=60, verbose=False):
    """Register device with DashBox server."""
    self.spinner = spinner

    sunnylink_dongle_id, comma_dongle_id = self._resolve_dongle_ids()

    if sunnylink_dongle_id not in (None, UNREGISTERED_SUNNYLINK_DONGLE_ID):
      # Already registered — but maybe missing pairing code
      if not self.params.get("SunnylinkPairingCode"):
        self._fetch_pairing_code(sunnylink_dongle_id)
      return sunnylink_dongle_id

    serial = self._resolve_serial()
    imei1, imei2 = self._resolve_imeis()
    imei = imei1 or imei2 or "unknown"

    _, __, public_key = BaseApi.get_key_pair()

    if not public_key:
      self._status_update("Public key not found, registering without it.")
      public_key = b""
    start_time = time.monotonic()
    backoff = 1
    while True:
      try:
        self._status_update("Registering device to DashBox...")
        resp = self._dash_post("devices/register", {
          "serial": serial,
          "imei": imei,
          "public_key": public_key.decode() if isinstance(public_key, bytes) else public_key,
        })

        if resp.status_code == 201:
          data = resp.json()
          sunnylink_dongle_id = data["device_id"]
          token = data["token"]
          self.params.put("SunnylinkToken", token)
          self._status_update(f"DashBox registered: {sunnylink_dongle_id}")

          # Fetch pairing code
          self._fetch_pairing_code(sunnylink_dongle_id)
          break
        else:
          raise Exception(f"Registration failed: {resp.status_code} {resp.text[:200]}")

      except Exception as e:
        if verbose:
          self._status_update(f"Retry in {backoff}s: {e}")
        backoff = min(backoff * 2, 60)
        time.sleep(backoff)

      if time.monotonic() - start_time > timeout:
        self._status_update(f"Giving up after {timeout}s")
        break

    self.params.put("DongleId", sunnylink_dongle_id or UNREGISTERED_SUNNYLINK_DONGLE_ID)
    self.params.put("LastSunnylinkPingTime", int(time.monotonic() * 1e9))

    self.spinner = None
    return sunnylink_dongle_id

  def _fetch_pairing_code(self, device_id):
    """Fetch a pairing code from DashBox server."""
    try:
      token = self.params.get("SunnylinkToken")
      if not token:
        self._status_update("No DashBox token, skipping")
        return
      resp = self._dash_post(f"devices/{device_id}/pair", {}, token=token)
      if resp.status_code == 200:
        data = resp.json()
        code = data.get("pairing_code", "")
        if code:
          self.params.put("SunnylinkPairingCode", code)
          self._status_update(f"Pairing code: {code}")
      else:
        self._status_update(f"Pairing failed: {resp.status_code} {resp.text[:100]}")
    except Exception as e:
      self._status_update(f"Failed to fetch pairing code: {e}")
