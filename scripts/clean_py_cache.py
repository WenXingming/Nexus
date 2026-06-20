"""Remove Python cache files and __pycache__ directories in this repository."""

from __future__ import annotations

import shutil
import os
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def make_writable_and_retry(function, path, _exc_info) -> None:
    try:
        os.chmod(path, stat.S_IWRITE)
        function(path)
    except PermissionError:
        print(f"warning: unable to remove {path}")


for cache_dir in ROOT.rglob("__pycache__"):
    shutil.rmtree(cache_dir, onexc=make_writable_and_retry)

for pyc_file in ROOT.rglob("*.pyc"):
    try:
        os.chmod(pyc_file, stat.S_IWRITE)
        pyc_file.unlink()
    except PermissionError:
        print(f"warning: unable to remove {pyc_file}")
