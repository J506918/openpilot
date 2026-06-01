"""
DashBox uploader — upload route logs to DashBox server.
Replaces sunnylink/uploader.py.
"""

import json
import os
import random
import time
import traceback
from collections.abc import Iterator

from openpilot.sunnypilot.sunnylink.api import SunnylinkApi  # retained for upload_url API shape
from openpilot.common.params import Params
from openpilot.system.hardware.hw import Paths
from openpilot.system.loggerd.xattr_cache import getxattr, setxattr
from openpilot.common.swaglog import cloudlog

try:
    from dashbox.api import DashboxApi, DASHBOX_HOST
except ImportError:
    from openpilot.sunnypilot.dashbox.api import DashboxApi, DASHBOX_HOST

UPLOAD_ATTR_NAME = 'user.dashbox.upload'
UPLOAD_ATTR_VALUE = b'1'


def listdir_by_creation(d: str) -> list[str]:
    if not os.path.isdir(d):
        return []
    try:
        paths = [f for f in os.listdir(d) if os.path.isdir(os.path.join(d, f))]
        paths = sorted(paths)
        return paths
    except OSError:
        cloudlog.exception("listdir_by_creation failed")
        return []


class DashBoxUploader:
    def __init__(self, dongle_id: str, root: str):
        self.dongle_id = dongle_id
        self.api = DashboxApi(dongle_id)
        self.root = root
        self.params = Params()

    def list_upload_files(self) -> Iterator[tuple[str, str, str]]:
        for logdir in listdir_by_creation(self.root):
            path = os.path.join(self.root, logdir)
            try:
                names = os.listdir(path)
            except OSError:
                continue

            if any(name.endswith(".lock") for name in names):
                continue

            for name in sorted(names):
                key = os.path.join(logdir, name)
                fn = os.path.join(path, name)
                try:
                    is_uploaded = getxattr(fn, UPLOAD_ATTR_NAME) == UPLOAD_ATTR_VALUE
                except OSError:
                    continue
                if is_uploaded:
                    continue
                yield name, key, fn

    def next_file_to_upload(self) -> tuple[str, str, str] | None:
        for item in self.list_upload_files():
            return item
        return None

    def step(self) -> bool | None:
        d = self.next_file_to_upload()
        if d is None:
            return None

        name, key, fn = d
        try:
            sz = os.path.getsize(fn)
        except OSError:
            return False

        if sz == 0:
            setxattr(fn, UPLOAD_ATTR_NAME, UPLOAD_ATTR_VALUE)
            return True

        cloudlog.event("dashbox_upload_start", key=key, fn=fn, sz=sz)

        try:
            upload_info = self.api.get_upload_url(key)
            url = upload_info['url']
            headers = upload_info.get('headers', {})

            import requests
            with open(fn, 'rb') as f:
                resp = requests.put(url, data=f, headers=headers, timeout=30)

            if resp.status_code in (200, 201):
                setxattr(fn, UPLOAD_ATTR_NAME, UPLOAD_ATTR_VALUE)
                cloudlog.event("dashbox_upload_success", key=key, sz=sz)
                return True
            else:
                cloudlog.event("dashbox_upload_failed", key=key, status=resp.status_code)
                return False
        except Exception as e:
            cloudlog.event("dashbox_upload_error", key=key, error=str(e))
            return False


def main():
    params = Params()
    dongle_id = params.get("DongleId")
    if not dongle_id:
        cloudlog.error("DashBox uploader: no dongle_id")
        return

    uploader = DashBoxUploader(dongle_id, Paths.log_root())
    while True:
        result = uploader.step()
        if result is None:
            time.sleep(5)
        elif result:
            time.sleep(0.1)
        else:
            time.sleep(random.uniform(1, 5))
