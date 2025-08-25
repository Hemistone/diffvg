"""
Dev-time safety: avoid corrupted .pyc after hard crashes.

When running scripts from the repo root, this disables bytecode writes by
default to prevent "bad marshal data" on subsequent runs if the process
segfaults mid-import. Opt out by setting DIFFVG_DEV_NOPYC=0.
"""
import os
import sys

if os.environ.get("DIFFVG_DEV_NOPYC", "1") != "0":
    sys.dont_write_bytecode = True

