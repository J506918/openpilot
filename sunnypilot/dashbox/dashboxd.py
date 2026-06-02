"""
DashBox WebSocket client — persistent connection to DashBox server.
Auth: serial + dongle_id, no token.
After server returns dongle_id (OK), device stores it and skips
registration on reconnect. Only WS auth failure triggers re-registration.
"""

import json
import time
import threading

import websocket

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.common.realtime import set_core_affinity

DASHBOX_WS_URL = "wss://8.136.28.140:8443/ws"
RECONNECT_DELAY = 5
PING_INTERVAL = 30


class DashboxDaemon:
    def __init__(self):
        self._params = Params()
        self._ws: websocket.WebSocket | None = None
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            dongle_id = self._params.get("DongleId") or ""
            if not dongle_id:
                if not self._reregister():
                    time.sleep(RECONNECT_DELAY)
                    continue
            self._connect()
            if self._running:
                time.sleep(RECONNECT_DELAY)

    def stop(self):
        self._running = False
        if self._ws:
            self._ws.close()

    def _connect(self):
        serial = self._read_serial()
        dongle_id = self._params.get("DongleId") or ""

        url = f"{DASHBOX_WS_URL}?serial={serial}&dongle_id={dongle_id}"
        cloudlog.info("DashBox WS connecting...")

        try:
            self._ws = websocket.create_connection(
                url,
                timeout=10,
                sslopt={"cert_reqs": 0},
            )
        except websocket.WebSocketBadStatusException as e:
            cloudlog.warning(f"DashBox WS: auth failed ({e.status_code}), clearing dongle_id")
            self._params.put("DongleId", "")
            return

        ping_thread = threading.Thread(target=self._ping_loop, daemon=True)
        ping_thread.start()

        # Auto-request pairing code so user can bind from app
        if not self._has_valid_pairing_code():
            self._request_pairing_code()

        while self._running:
            try:
                msg = self._ws.recv()
                if msg:
                    self._handle_message(msg)
            except websocket.WebSocketTimeoutException:
                continue
            except Exception:
                cloudlog.exception("DashBox WS read error")
                break

    def _reregister(self):
        """Register with server. Returns True on success, False on failure."""
        cloudlog.info("DashBox: no dongle_id, registering...")
        try:
            from sunnypilot.dashbox.api import register_dashbox
            new_id = register_dashbox()
            cloudlog.info(f"DashBox: registered as {new_id}")
            return True
        except Exception:
            cloudlog.exception("DashBox: registration failed")
            return False

    def _has_valid_pairing_code(self) -> bool:
        """Check if we already have a non-expired pairing code cached."""
        code = self._params.get("SunnylinkPairingCode") or ""
        ts_str = self._params.get("DashboxPairingCodeTs") or "0"
        try:
            ts = float(ts_str)
        except ValueError:
            return False
        if not code or time.time() - ts > 840:  # 14 min, server TTL is 15
            return False
        return True

    def _request_pairing_code(self):
        """Request pairing code from server via WS RPC."""
        try:
            req = json.dumps({
                "jsonrpc": "2.0",
                "method": "generate_pairing_code",
                "id": int(time.time() * 1000),
            })
            self._ws.send(req)
            cloudlog.info("DashBox: requested pairing code")
        except Exception:
            cloudlog.exception("DashBox: failed to request pairing code")

    def _ping_loop(self):
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
    def _read_serial() -> str:
        try:
            with open("/data/params/d/HardwareSerial", "r") as f:
                return f.read().strip()
        except Exception:
            return "unknown"

    def _handle_message(self, raw: str):
        """Handle incoming JSON-RPC messages from server."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        method = msg.get("method")
        if method == "saveParams":
            params = msg.get("params", {}).get("params", {})
            cloudlog.info(f"DashBox: received {len(params)} params from server")
            for key, value in params.items():
                self._params.put(key, str(value))

        elif method == "pong":
            pass

        # Handle RPC responses (have "result" or "error", no "method")
        elif "result" in msg and isinstance(msg["result"], dict):
            result = msg["result"]
            if "pairing_code" in result:
                code = result["pairing_code"]
                self._params.put("DashboxPairingCodeTs", str(time.time()))
                # Store via storage module so the pairing dialog can read it
                try:
                    from sunnypilot.dashbox.storage import put as storage_put
                    storage_put("SunnylinkPairingCode", code)
                except Exception:
                    self._params.put("SunnylinkPairingCode", code)
                cloudlog.info(f"DashBox: pairing code {code} stored")


def main():
    params = Params()

    if not params.get_bool("SunnylinkEnabled"):
        return

    set_core_affinity([0, 1, 2, 3])
    daemon = DashboxDaemon()
    daemon.run()


if __name__ == "__main__":
    main()
