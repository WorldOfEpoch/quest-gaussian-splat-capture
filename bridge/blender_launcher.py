"""Open a verified project scan in a fresh local Blender 5.3 window."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

from .dashboard import dashboard_current_model, verified_job


class BlenderLaunchError(Exception):
    pass


def _blender_executable() -> Path:
    configured = os.environ.get("QUEST_SPLAT_BLENDER_EXE")
    choices = [Path(configured)] if configured else []
    installed = shutil.which("blender")
    if installed:
        choices.append(Path(installed))
    for candidate in choices:
        if not candidate.is_file():
            continue
        try:
            result = subprocess.run([str(candidate), "--version"], capture_output=True,
                                    text=True, timeout=15, check=True)
        except (OSError, subprocess.SubprocessError):
            continue
        version = re.search(r"\bBlender (\d+)\.(\d+)", result.stdout)
        if version and (int(version.group(1)), int(version.group(2))) >= (5, 3):
            return candidate
    raise BlenderLaunchError("Blender 5.3 or newer was not found on this PC")


def _current_splat(root: Path, project_id: str) -> Path:
    uuid.UUID(project_id)
    model = dashboard_current_model(root, project_id)
    if not model:
        raise BlenderLaunchError("This project has no verified Gaussian splat yet")
    if model["source"] == "saved_baseline":
        found = verified_job(root, project_id, model["id"])
        if not found:
            raise BlenderLaunchError("The saved Gaussian splat is no longer verified")
        return found[1]
    if model["source"] != "online":
        raise BlenderLaunchError("This model cannot be imported as a Gaussian splat")

    # Online snapshots can be overwritten while Blender starts. Pin the exact
    # verified bytes so the native importer sees an immutable revision.
    session_id = str(uuid.UUID(model["session_id"]))
    expected = model["id"].rsplit(":", 1)[-1]
    source = root / "projects" / project_id / "online" / session_id / "current.ply"
    if source.is_symlink() or not source.is_file() or source.stat().st_size > 256 * 1024 * 1024:
        raise BlenderLaunchError("The current online snapshot is unavailable")
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise BlenderLaunchError("The online snapshot changed; refresh and try again")
    directory = root / "projects" / project_id / "blender_snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{expected}.ply"
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        temporary = directory / f".{uuid.uuid4()}.tmp"
        try:
            temporary.write_bytes(content)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return target


def open_current_splat_in_blender(root: Path, project_id: str) -> int:
    source = _current_splat(root, project_id)
    return _launch_splat(root, project_id, source)


def open_saved_splat_in_blender(root: Path, project_id: str, job_id: str) -> int:
    found = verified_job(root, project_id, job_id)
    if not found:
        raise BlenderLaunchError("The saved Gaussian splat is unavailable or unverified")
    return _launch_splat(root, project_id, found[1])


def _launch_splat(root: Path, project_id: str, source: Path) -> int:
    blender = _blender_executable()
    script = Path(__file__).resolve().parents[1] / "blender_addon" / "open_scan.py"
    if not script.is_file():
        raise BlenderLaunchError("Blender scan importer is missing")
    log_path = root / "blender-open.log"
    try:
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                [str(blender), "--factory-startup", "--python", str(script), "--",
                 str(source), project_id],
                cwd=str(script.parent), stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, close_fds=True,
            )
    except OSError as exc:
        raise BlenderLaunchError("Blender could not be started") from exc
    return process.pid
