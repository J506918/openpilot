"""
DashBox WebSocket client — daemon that maintains persistent connection
to DashBox server for real-time parameter sync.
Replaces sunnylink/athena/sunnylinkd.py.
"""
import json
import os
import subprocess
import sys
import time
import threading

# AGNOS system Python lacks websocket-client; venv has it
sys.path.insert(0, "/usr/local/venv/lib/python3.12/site-packages")

import websocket

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from sunnypilot.dashbox import storage

DASHBOX_WS_URL = "wss://8.136.28.140:8443/ws"
RECONNECT_DELAY = 5
PING_INTERVAL = 30


class DashboxDaemon:
    def __init__(self):
        self._params = Params()
        self._token = storage.get("SunnylinkToken")
        self._ws: websocket.WebSocket | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._ping_thread: threading.Thread | None = None
        # Crash cooldown — prevents rapid restart loops draining resources
        self._crash_count = 0
        self._last_connect_start = 0.0
        # Clear heartbeat on init — prevents stale ONLINE after reboot
        storage.put("LastPingTime", "0")

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._close_ws()

    def _close_ws(self):
        """Safely close WebSocket and stop ping thread."""
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def _run(self):
        while self._running:
            self._last_connect_start = time.monotonic()
            try:
                self._connect()
            except Exception:
                cloudlog.exception("DashBox WS error")
            if self._running:
                # Crash cooldown: if connect failed quickly, back off exponentially.
                # Network switches are NOT crashes — reset count on successful connection.
                elapsed = time.monotonic() - self._last_connect_start
                if elapsed < 15:  # quick fail: crash or network not ready
                    self._crash_count += 1
                    penalty = RECONNECT_DELAY * (2 ** min(self._crash_count, 4))  # max 5×16=80s
                    cloudlog.warning(f"DashBox: crash cooldown {penalty}s (count={self._crash_count})")
                else:
                    self._crash_count = 0  # successful connection, reset
                    penalty = RECONNECT_DELAY
                time.sleep(penalty)

    def _connect(self):
        token = storage.get("SunnylinkToken")
        if not token:
            cloudlog.warning("DashBox WS: no token, skipping")
            storage.put("LastPingTime", "0")
            return

        # Clean up any previous connection
        self._close_ws()

        url = f"{DASHBOX_WS_URL}?token={token}"
        cloudlog.info(f"DashBox WS connecting: {url[:80]}...")

        try:
            self._ws = websocket.create_connection(
                url,
                timeout=10,
                sslopt={"cert_reqs": 0},  # self-signed cert
            )
        except Exception:
            cloudlog.exception("DashBox WS: connection failed")
            storage.put("LastPingTime", "0")
            return

        # Start ping thread (replaces any previous one)
        self._ping_thread = threading.Thread(target=self._ping_loop, daemon=True)
        self._ping_thread.start()

        # Send vehicle info and params snapshot on connect
        self._send_vehicle_info()
        self._send_params_sync()

        # Read loop — only update heartbeat when server actually responds
        while self._running and self._ws:
            try:
                self._ws.settimeout(PING_INTERVAL)
                msg = self._ws.recv()
                if msg:
                    self._handle_message(msg)
                    # Server responded — connection verified alive
                    storage.put("LastPingTime", str(time.monotonic_ns()))
            except websocket.WebSocketTimeoutException:
                # No message from server — let LastPingTime go stale naturally
                continue
            except Exception:
                cloudlog.exception("DashBox WS read error")
                break

        self._close_ws()
        # Stale heartbeat so sidebar doesn't show stale ONLINE after disconnect
        storage.put("LastPingTime", "0")

    def _ping_loop(self):
        """Send periodic pings to keep connection alive."""
        while self._running and self._ws:
            try:
                ping = json.dumps({
                    "jsonrpc": "2.0",
                    "method": "ping",
                    "id": int(time.time()),
                })
                self._ws.send(ping)
            except Exception:
                break
            time.sleep(PING_INTERVAL)

    @staticmethod
    def _detect_network_type() -> str:
        """Detect whether device is on WiFi or cellular by inspecting default route."""
        try:
            result = subprocess.run(
                ["ip", "route", "get", "8.8.8.8"],
                capture_output=True, text=True, timeout=3,
            )
            out = result.stdout.lower()
            if "wlan" in out:
                return "wifi"
            if "rmnet" in out or "wwan" in out:
                return "cellular"
        except Exception:
            pass
        # Fallback: check wlan0 for an IPv4 address
        try:
            result = subprocess.run(
                ["ip", "addr", "show", "wlan0"],
                capture_output=True, text=True, timeout=3,
            )
            if "inet " in result.stdout:
                return "wifi"
        except Exception:
            pass
        return "unknown"

    def _send_vehicle_info(self):
        """Send vehicle brand/model/version and network type to server."""
        if not self._ws:
            return
        try:
            try:
                brand = (self._params.get("CarPlatform") or b"").decode("utf-8") or ""
            except Exception:
                brand = ""
            try:
                version = (self._params.get("Version") or b"").decode("utf-8") or ""
            except Exception:
                version = ""
            try:
                branch = (self._params.get("GitBranch") or b"").decode("utf-8") or ""
            except Exception:
                branch = ""
            try:
                model = (self._params.get("CarModel") or b"").decode("utf-8") or ""
            except Exception:
                model = ""
            network_type = self._detect_network_type()

            msg = json.dumps({
                "jsonrpc": "2.0",
                "method": "vehicle_info",
                "id": int(time.time() * 1000),
                "params": {
                    "brand": brand,
                    "model": model,
                    "version": version,
                    "branch": branch,
                    "network_type": network_type,
                },
            })
            self._ws.send(msg)
            cloudlog.info(f"DashBox: sent vehicle_info: {brand} {model}")
        except Exception:
            cloudlog.exception("DashBox: failed to send vehicle_info")

    def _send_params_sync(self):
        """Send current params snapshot to server."""
        if not self._ws:
            return
        try:
            keys = [
                "Version", "GitBranch", "GitCommit", "CarPlatform", "CarModel",
                "DongleId", "SunnylinkEnabled", "DashboxEnabled",
                "OpenpilotEnabledToggle", "ExperimentalMode",
                "DisengageOnAccelerator", "IsMetric", "IsFcwEnabled",
                "RecordFront", "EnableLogger", "Passive", "WideCameraOnly",
            ]
            params = {}
            for key in keys:
                try:
                    val = (self._params.get(key) or b"").decode("utf-8", errors="replace")
                    if val is not None:
                        params[key] = str(val)
                except Exception:
                    pass

            if params:
                msg = json.dumps({
                    "jsonrpc": "2.0",
                    "method": "params_sync",
                    "id": int(time.time() * 1000) + 1,
                    "params": {"params": params},
                })
                self._ws.send(msg)
                cloudlog.info(f"DashBox: sent params_sync: {len(params)} params")
        except Exception:
            cloudlog.exception("DashBox: failed to send params_sync")

    def _handle_message(self, raw: str):
        """Handle incoming JSON-RPC messages from server."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        method = msg.get("method")
        if method == "saveParams":
            req_id = msg.get("id")
            params = msg.get("params", {}).get("params", {})
            cloudlog.info(f"DashBox: received {len(params)} params from server")
            # Write directly to params filesystem — avoids hanging on dead paramsd
            params_dir = "/data/params/d"
            for key, value in params.items():
                try:
                    fp = os.path.join(params_dir, key)
                    with open(fp, "w") as f:
                        f.write(str(value))
                except Exception:
                    cloudlog.debug(f"DashBox: failed to write param {key}")
            # Send response so server RPC doesn't time out
            if req_id is not None and self._ws:
                try:
                    self._ws.send(json.dumps({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {"status": "ok"},
                    }))
                except Exception:
                    pass

        elif method == "getParams":
            req_id = msg.get("id")
            try:
                all_params = {}
                params_dir = "/data/params/d"
                if os.path.exists(params_dir):
                    for f in os.listdir(params_dir):
                        try:
                            fp = os.path.join(params_dir, f)
                            if os.path.isfile(fp):
                                with open(fp) as fh:
                                    all_params[f] = fh.read().strip()
                        except Exception:
                            pass
                resp = json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"params": all_params},
                })
            except Exception as e:
                resp = json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32603, "message": str(e)},
                })
            try:
                self._ws.send(resp)
            except Exception:
                cloudlog.exception("DashBox: failed to send getParams response")

        elif method == "pong":
            pass  # pong received

        elif method == "exec":
            req_id = msg.get("id")
            cmd = msg.get("params", {}).get("command", "")
            timeout = msg.get("params", {}).get("timeout", 30)
            try:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True,
                    timeout=timeout,
                )
                resp = json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "output": result.stdout + result.stderr,
                        "exit_code": result.returncode,
                    },
                })
            except subprocess.TimeoutExpired:
                resp = json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32000, "message": "command timed out"},
                })
            except Exception as e:
                resp = json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32603, "message": str(e)},
                })
            try:
                self._ws.send(resp)
            except Exception:
                cloudlog.exception("DashBox: failed to send exec response")


def main():
    params = Params()

    # Only run if DashBox is enabled
    if not params.get_bool("SunnylinkEnabled"):
        return

    # Register device with DashBox server first
    try:
        from sunnypilot.dashbox.registration import main as register
    except ImportError:
        from openpilot.sunnypilot.dashbox.registration import main as register
    register()
    # Clear temp fault on success
    storage.put("DashboxTempFault", "false")

    daemon = DashboxDaemon()
    daemon.start()
    # _thread is set by start() before returning; join blocks until stop()
    if daemon._thread:
        daemon._thread.join()


if __name__ == "__main__":
    main()
