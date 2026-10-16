"""Translate two removed COLMAP 3.x GPU flags for installed COLMAP 4.x."""

from __future__ import annotations

import os
import subprocess
import sys


RENAMED = {
    "--SiftExtraction.use_gpu": "--FeatureExtraction.use_gpu",
    "--SiftMatching.use_gpu": "--FeatureMatching.use_gpu",
}


def main() -> int:
    executable = os.environ.get("QUEST_SPLAT_COLMAP_EXE")
    if not executable or not os.path.isfile(executable):
        print("QUEST_SPLAT_COLMAP_EXE must point to the installed COLMAP executable", file=sys.stderr)
        return 2
    args = [RENAMED.get(arg, arg) for arg in sys.argv[1:]]
    return subprocess.call([executable, *args])


if __name__ == "__main__":
    raise SystemExit(main())
