"""
DashBox WebSocket client — persistent connection to DashBox server.
Auth: serial + dongle_id, no token.
"""
import json
import os
import select
import socket
import subprocess
import time

import websocket

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.common.realtime import set_core_affinity

DASHBOX_WS_URL = "wss://8.136.28.140:8443/ws"
RECONNECT_DELAY = 2
NOTIFY_SOCK = "/tmp/dashbox_notify.sock"
UI_SOCK = "/tmp/dashbox_ui.sock"


class DashboxDaemon:
    def __init__(self):
        self._params = Params()
        self._ws: websocket.WebSocket | None = None
        self._notify_sock: socket.socket | None = None
        self._running = False
        self._crash_count = 0
        self._last_connect_start = 0.0
        self._last_ping_time = 0.0
        # Clear online flag on init — prevents stale ONLINE after crash/reboot
        self._write_file("/data/params/d/DashboxOnline", "0")

    def run(self):
        self._running = True
        while self._running:
            self._last_connect_start = time.monotonic()
            try:
                self._connect()
            except Exception:
                cloudlog.exception("DashBox WS error")
            # Crash cooldown — exponential backoff on quick failures
            elapsed = time.monotonic() - self._last_connect_start
            if elapsed < 15:
                self._crash_count += 1
                penalty = RECONNECT_DELAY * (2 ** min(self._crash_count, 4))
                cloudlog.warning(f"DashBox: crash cooldown {penalty}s (count={self._crash_count})")
            else:
                self._crash_count = 0
                penalty = RECONNECT_DELAY
            if self._running:
                time.sleep(penalty)

    def stop(self):
        self._running = False
        self._close_ws()

    def _close_ws(self):
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._notify_sock:
            try:
                self._notify_sock.close()
            except Exception:
                pass
            self._notify_sock = None
            try:
                os.unlink(NOTIFY_SOCK)
            except OSError:
                pass

    def _connect(self):
        self._close_ws()

        serial = self._read_serial()
        dongle_id = self._params.get("DongleId") or ""
        url = f"{DASHBOX_WS_URL}?serial={serial}&dongle_id={dongle_id}"
        cloudlog.info("DashBox WS connecting...")

        try:
            self._ws = websocket.create_connection(
                url, timeout=10, sslopt={"cert_reqs": 0},
                ping_interval=5, ping_timeout=20,
            )
        except Exception:
            cloudlog.exception("DashBox WS: connection failed")
            return

        # Server may send fix_dongle_id if ID is missing or mismatched
        try:
            self._ws.settimeout(3)
            first_msg = self._ws.recv()
            data = json.loads(first_msg)
            if data.get("method") == "fix_dongle_id":
                new_id = data["params"]["dongle_id"]
                cloudlog.info(f"DashBox: server assigned dongle_id={new_id}")
                self._params.put("DongleId", new_id)
                self._ws.close()
                self._ws = None
                return
        except Exception:
            pass  # No fix message, normal connection

        cloudlog.info("DashBox WS: connected")

        # Push state snapshot on connect
        self._send_vehicle_info()
        self._send_params_sync()

        # Set up notify socket — UI toggles write param changes here
        try:
            os.unlink(NOTIFY_SOCK)
        except OSError:
            pass
        self._notify_sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._notify_sock.bind(NOTIFY_SOCK)
        self._notify_sock.setblocking(False)

        # Short recv timeout — prevents select loop from blocking on
        # control frames (ping) before data frames (RPC).
        self._ws.settimeout(1)

        # Select loop: WebSocket + notify socket
        while self._running and self._ws:
            try:
                ws_fd = self._ws.sock.fileno()
                r, _, _ = select.select([ws_fd, self._notify_sock], [], [], 5)
                now = time.monotonic()

                if ws_fd in r:
                    self._last_ping_time = now  # Server sent data or ping – reset heartbeat
                    try:
                        msg = self._ws.recv()
                        if msg:
                            self._handle_message(msg)
                    except websocket.WebSocketTimeoutException:
                        pass

                if self._notify_sock in r:
                    updates = {}
                    try:
                        while True:
                            data = self._notify_sock.recv(4096)
                            for line in data.decode().strip().split("\n"):
                                if "=" in line:
                                    k, v = line.split("=", 1)
                                    updates[k] = v
                    except BlockingIOError:
                        pass
                    if updates:
                        self._send_params_push(updates)

                # Heartbeat: DashboxOnline = 1 if ping within 20s, 0 otherwise
                online = (now - self._last_ping_time) < 20
                new_state = "1" if online else "0"
                current = self._read_dashbox_online()
                if new_state != current:
                    self._write_file("/data/params/d/DashboxOnline", new_state)
                    cloudlog.debug(f"DashBox: heartbeat -> {new_state}")

            except websocket.WebSocketTimeoutException:
                continue
            except Exception:
                cloudlog.exception("DashBox WS read error")
                break

        self._write_file("/data/params/d/DashboxOnline", "0")
        self._close_ws()
        cloudlog.info("DashBox WS: disconnected")

    # ── helpers ─────────────────────────────────────────────────

    @staticmethod
    def _write_file(path, value):
        try:
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                f.write(value)
            os.rename(tmp, path)
        except Exception:
            pass

    @staticmethod
    def _read_dashbox_online():
        try:
            with open("/data/params/d/DashboxOnline", "r") as f:
                return f.read().strip()
        except Exception:
            return "0"

    @staticmethod
    def _read_serial():
        try:
            with open("/data/params/d/HardwareSerial", "r") as f:
                return f.read().strip()
        except Exception:
            return "unknown"

    @staticmethod
    def _detect_network_type():
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

    # ── outgoing RPC ────────────────────────────────────────────

    def _send_vehicle_info(self):
        if not self._ws:
            return
        try:
            brand = model = ""
            bundle_path = os.path.join("/data/params/d", "CarPlatformBundle")
            try:
                with open(bundle_path, "r") as f:
                    bundle = json.load(f)
                    brand = bundle.get("make", "") or bundle.get("brand", "")
                    model = bundle.get("model", "")
            except Exception:
                pass
            if not brand:
                try:
                    brand = (self._params.get("CarPlatform") or b"").decode("utf-8") or ""
                except Exception:
                    pass
            if not model:
                try:
                    model = (self._params.get("CarModel") or b"").decode("utf-8") or ""
                except Exception:
                    pass
            version = branch = ""
            for key, target in [("Version", "version"), ("GitBranch", "branch")]:
                try:
                    fp = os.path.join("/data/params/d", key)
                    with open(fp, "r") as f:
                        val = f.read().strip()
                    if target == "version":
                        version = val
                    else:
                        branch = val
                except Exception:
                    pass
            network_type = self._detect_network_type()
            msg = json.dumps({
                "jsonrpc": "2.0",
                "method": "vehicle_info",
                "id": int(time.time() * 1000),
                "params": {
                    "brand": brand, "model": model,
                    "version": version, "branch": branch,
                    "network_type": network_type,
                },
            })
            self._ws.send(msg)
            cloudlog.info(f"DashBox: sent vehicle_info: {brand} {model}")
        except Exception:
            cloudlog.exception("DashBox: failed to send vehicle_info")

    def _send_params_sync(self):
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
                    if val:
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

    def _send_params_push(self, updates):
        if not self._ws:
            return
        try:
            msg = json.dumps({
                "jsonrpc": "2.0",
                "method": "params_sync",
                "params": {"params": updates},
            })
            self._ws.send(msg)
            cloudlog.info(f"DashBox: push params: {list(updates.keys())}")
        except Exception:
            cloudlog.exception("DashBox: failed to push params")

    # ── incoming message handling ───────────────────────────────

    def _handle_message(self, raw):
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        method = msg.get("method")
        if method == "saveParams":
            req_id = msg.get("id")
            params = msg.get("params", {}).get("params", {})
            cloudlog.info(f"DashBox: received {len(params)} params from server")
            params_dir = "/data/params/d"
            changed_keys = []
            for key, value in params.items():
                try:
                    fp = os.path.join(params_dir, key)
                    tmp = fp + ".tmp"
                    with open(tmp, "w") as f:
                        f.write(str(value))
                    os.rename(tmp, fp)
                    changed_keys.append(key)
                except Exception:
                    cloudlog.debug(f"DashBox: failed to write param {key}")
            # Notify UI via Unix socket so widgets refresh
            if changed_keys:
                try:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
                    s.sendto("\n".join(changed_keys).encode(), UI_SOCK)
                    s.close()
                except Exception:
                    pass
            # Reply to server RPC
            if req_id is not None and self._ws:
                try:
                    self._ws.send(json.dumps({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {"status": "ok"},
                    }))
                except Exception:
                    pass
            # Re-push params so server forwards to App
            self._send_params_sync()

        elif method == "getParams":
            req_id = msg.get("id")
            try:
                all_params = {}
                params_dir = "/data/params/d"
                if os.path.exists(params_dir):
                    for f in os.listdir(params_dir):
                        if f.endswith(".tmp") or f.endswith(".lock"):
                            continue
                        try:
                            fp = os.path.join(params_dir, f)
                            if os.path.isfile(fp):
                                with open(fp, "rb") as fh:
                                    raw_bytes = fh.read()
                                    try:
                                        all_params[f] = raw_bytes.decode("utf-8")
                                    except UnicodeDecodeError:
                                        all_params[f] = raw_bytes.decode("utf-8", errors="replace")
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
            pass

        elif method == "exec":
            req_id = msg.get("id")
            cmd = msg.get("params", {}).get("command", "")
            timeout_val = msg.get("params", {}).get("timeout", 30)
            try:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True,
                    timeout=timeout_val,
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
    if not params.get_bool("SunnylinkEnabled"):
        return
    set_core_affinity([0, 1, 2, 3])
    daemon = DashboxDaemon()
    daemon.run()


if __name__ == "__main__":
    main()
