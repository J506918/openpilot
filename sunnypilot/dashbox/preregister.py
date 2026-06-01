"""
DashBox pre-registration — runs before UI starts during manager_init().
Similar timing to comma's registration flow, but for DashBox server.
Has a hard timeout (default 180s) so it never blocks boot indefinitely.
"""
import threading

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.system.athena.registration import UNREGISTERED_DONGLE_ID

try:
    from sunnypilot.dashbox import storage
except ImportError:
    from openpilot.sunnypilot.dashbox import storage


def dashbox_preregister(timeout: float = 180.0) -> str | None:
    """
    Register device with DashBox server before UI starts.

    - If already registered (has token + valid DongleId), returns immediately.
    - Otherwise attempts registration with a timeout.
    - Never raises — always returns device_id or None.

    Args:
        timeout: Maximum seconds to wait (default 180 = 3 minutes)
    
    Returns:
        DashBox device_id on success, None on timeout/failure.
    """
    params = Params()

    # ── Already registered? ───────────────────────────────────────
    dongle_id = params.get("DongleId")
    if dongle_id and dongle_id != UNREGISTERED_DONGLE_ID:
        token = storage.get("SunnylinkToken")
        if token:
            cloudlog.info(f"DashBox preregister: already registered ({dongle_id})")
            return dongle_id

    cloudlog.info(f"DashBox preregister: starting (timeout={timeout}s)")

    result: list[str | None] = [None]

    def _do_register() -> None:
        try:
            # Use existing register_dashbox which handles retry + token + pairing
            try:
                from sunnypilot.dashbox.api import register_dashbox
            except ImportError:
                from openpilot.sunnypilot.dashbox.api import register_dashbox
            device_id = register_dashbox()
            result[0] = device_id
        except Exception:
            cloudlog.exception("DashBox preregister error")

    t = threading.Thread(target=_do_register, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if result[0]:
        cloudlog.info(f"DashBox preregister: success ({result[0]})")
        return result[0]

    cloudlog.warning(f"DashBox preregister: timed out after {timeout}s")
    return None
