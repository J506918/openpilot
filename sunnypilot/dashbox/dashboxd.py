"""
DashBox WebSocket client — daemon that maintains persistent connection
to DashBox server for real-time parameter sync.
Replaces sunnylink/athena/sunnylinkd.py.
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
        self._token = self._params.get("DashboxToken")
        self._ws: websocket.WebSocket | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._ws:
            self._ws.close()

    def _run(self):
        while self._running:
            try:
                self._connect()
            except Exception:
                cloudlog.exception("DashBox WS error")
            if self._running:
                time.sleep(RECONNECT_DELAY)

    def _connect(self):
        token = self._params.get("DashboxToken")
        if not token:
            cloudlog.warning("DashBox WS: no token, triggering re-registration")
            self._clear_and_reregister()
            return

        url = f"{DASHBOX_WS_URL}?token={token}"
        cloudlog.info(f"DashBox WS connecting...")

        try:
            self._ws = websocket.create_connection(
                url,
                timeout=10,
                sslopt={"cert_reqs": 0},
            )
        except websocket.WebSocketBadStatusException as e:
            cloudlog.warning(f"DashBox WS: auth failed ({e.status_code}), re-registering")
            self._clear_and_reregister()
            return

        ping_thread = threading.Thread(target=self._ping_loop, daemon=True)
        ping_thread.start()

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

    def _clear_and_reregister(self):
        """Clear stored credentials and trigger fresh registration."""
        cloudlog.info("DashBox: clearing stored credentials for re-registration")
        self._params.delete("DashboxDongleId")
        self._params.delete("DashboxToken")
        # Import here to avoid circular dependency
        try:
            from dashbox.api import register_dashbox
            new_id = register_dashbox()
            cloudlog.info(f"DashBox: re-registered as {new_id}")
        except Exception:
            cloudlog.exception("DashBox: re-registration failed")

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


def main():
    set_core_affinity([0, 1, 2, 3])
    daemon = DashboxDaemon()
    daemon._connect()


if __name__ == "__main__":
    main()
