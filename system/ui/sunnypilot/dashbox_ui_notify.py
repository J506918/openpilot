"""DashBox UI notify — real-time widget refresh from App param changes."""
import os
import select
import socket

NOTIFY_SOCK = "/tmp/dashbox_ui.sock"

_registry = {}  # param_key -> widget with refresh_from_params()
_sock = None


def register(key: str, widget):
    """Called by ToggleSP/OptionControlSP in __init__."""
    _registry[key] = widget


def init():
    """Create Unix datagram socket for dashboxd notifications."""
    global _sock
    try:
        os.unlink(NOTIFY_SOCK)
    except OSError:
        pass
    _sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    _sock.bind(NOTIFY_SOCK)
    _sock.setblocking(False)


def process_pending():
    """Call each frame. Zero-cost when no notifications pending."""
    if _sock is None:
        return
    try:
        r, _, _ = select.select([_sock], [], [], 0)
        if _sock not in r:
            return
        while True:
            try:
                data = _sock.recv(4096)
                for key in data.decode().strip().split("\n"):
                    key = key.strip()
                    if key and key in _registry:
                        _registry[key].refresh_from_params()
            except BlockingIOError:
                break
    except Exception:
        pass
