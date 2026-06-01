"""
DashBox persistent storage — bypasses C++ params whitelist.
Stores data in a JSON file at /data/dashbox_state.json,
completely independent of the openpilot params system.
"""
import json
import os
import threading

from openpilot.common.swaglog import cloudlog

STORAGE_PATH = "/data/dashbox_state.json"

_lock = threading.Lock()

# Map storage keys → params keys for settings UI sync
_PARAMS_SYNC_MAP = {
    "SunnylinkPairingCode": "SunnylinkPairingCode",
}


def _sync_to_params(key: str, value: str) -> None:
    """Sync a storage key to the corresponding params key for settings UI."""
    params_key = _PARAMS_SYNC_MAP.get(key)
    if not params_key:
        return
    try:
        from openpilot.common.params import Params
        Params().put(params_key, value)
    except Exception:
        pass  # params may reject during early boot


def _load() -> dict:
    """Load full state dict. Returns {} if file doesn't exist."""
    try:
        if os.path.exists(STORAGE_PATH):
            with open(STORAGE_PATH, "r") as f:
                return json.load(f)
    except Exception as e:
        cloudlog.warning(f"DashBox storage load error: {e}")
    return {}


def _save(state: dict) -> None:
    """Atomically write state dict to disk."""
    try:
        os.makedirs(os.path.dirname(STORAGE_PATH), exist_ok=True)
        tmp = STORAGE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.rename(tmp, STORAGE_PATH)  # atomic on same filesystem
    except Exception as e:
        cloudlog.error(f"DashBox storage save error: {e}")


def get(key: str) -> str:
    """Get a value. Returns empty string if key not found."""
    with _lock:
        state = _load()
        return state.get(key, "")


def put(key: str, value: str) -> None:
    """Put a value. Also syncs DashBox keys to params for settings UI display."""
    with _lock:
        state = _load()
        state[key] = value
        _save(state)
    # Sync DashBox keys to params for settings UI
    _sync_to_params(key, value)


def remove(key: str) -> None:
    """Remove a key."""
    with _lock:
        state = _load()
        state.pop(key, None)
        _save(state)


def get_all() -> dict:
    """Return all stored key-value pairs."""
    with _lock:
        return dict(_load())
