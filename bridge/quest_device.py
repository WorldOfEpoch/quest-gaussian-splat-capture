"""Explicit local Quest wake and wear-sensor controls shared by dashboard and CLI."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path


class QuestDeviceError(Exception):
    pass


_device_lock = threading.Lock()
ACTIONS = {"status", "wake", "keep-awake", "wear-sensor"}


def _run(executable: Path, arguments: list[str], *, adb_path: Path) -> str:
    env = dict(os.environ, HZDB_ADB_PATH=str(adb_path))
    try:
        result = subprocess.run(
            [str(executable), *arguments], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=15, check=True,
            env=env, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except subprocess.TimeoutExpired as error:
        raise QuestDeviceError("The Quest did not respond. Check the USB cable and try Wake headset again.") from error
    except (OSError, subprocess.CalledProcessError) as error:
        raise QuestDeviceError("The device command failed. Check the Quest connection and USB debugging authorization.") from error
    return result.stdout


def _tools(root: Path) -> tuple[Path, Path | None, str | None]:
    path = root / "device-tools.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (ValueError, OSError) as error:
        raise QuestDeviceError("The PC device-tool configuration could not be read.") from error
    if not isinstance(config, dict):
        raise QuestDeviceError("The PC device-tool configuration is invalid.")
    adb = config.get("adb_path") or os.environ.get("HZDB_ADB_PATH") or shutil.which("adb")
    meta = config.get("meta_cli_path") or shutil.which("metavr") or shutil.which("hzdb")
    if not isinstance(adb, str) or not Path(adb).is_file():
        raise QuestDeviceError("Set up the local ADB path in device-tools.json before using headset controls.")
    meta_path = Path(meta) if isinstance(meta, str) and Path(meta).is_file() else None
    serial = config.get("device_id")
    if serial is not None and (not isinstance(serial, str) or not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", serial)):
        raise QuestDeviceError("The configured Quest device ID is invalid.")
    return Path(adb), meta_path, serial


def _select_device(output: str, configured: str | None) -> str:
    rows = [line.split() for line in output.splitlines()]
    rows = [row for row in rows if len(row) >= 2 and row[1] in {"device", "offline", "unauthorized"}]
    if configured:
        matches = [row for row in rows if row[0] == configured]
        if not matches:
            raise QuestDeviceError("The configured Quest is not connected. Connect its USB cable.")
        if matches[0][1] != "device":
            raise QuestDeviceError("The Quest is offline or awaiting USB debugging approval inside the headset.")
        return configured
    matches = [row[0] for row in rows if row[1] == "device" and any(part.startswith("model:Quest") for part in row[2:])]
    if len(matches) != 1:
        raise QuestDeviceError("Connect exactly one authorized Quest, or select its device ID in device-tools.json.")
    return matches[0]


def _state(adb: Path, serial: str) -> dict:
    power = _run(adb, ["-s", serial, "shell", "dumpsys", "power"], adb_path=adb)
    display = _run(adb, ["-s", serial, "shell", "dumpsys", "display"], adb_path=adb)
    battery = _run(adb, ["-s", serial, "shell", "dumpsys", "battery"], adb_path=adb)
    def field(pattern: str, source: str):
        match = re.search(pattern, source)
        return match.group(1) if match else None
    level = field(r"(?m)^\s*level:\s*(\d+)", battery)
    return {"wakefulness": field(r"\bmWakefulness=(\w+)", power),
            "display_on": bool(re.search(r"\bmScreenState=ON\b", display)),
            "battery_percent": int(level) if level else None,
            "last_sleep_reason": field(r"\bmLastSleepReason=([^\s]+)", power)}


def perform_device_action(root: Path, action: str) -> dict:
    if action not in ACTIONS:
        raise QuestDeviceError("Unsupported headset action.")
    # Explicit user commands only: never poll-wake the headset, toggle power,
    # restart ADB, reboot, launch apps, or change the controller/boundary state.
    with _device_lock:
        adb, meta, configured = _tools(root)
        serial = _select_device(_run(adb, ["devices", "-l"], adb_path=adb), configured)
        if action in {"keep-awake", "wear-sensor"}:
            if meta is None:
                raise QuestDeviceError("Meta VR CLI is required to change the wear sensor.")
            flag = "--disable" if action == "keep-awake" else "--enable"
            _run(meta, ["-d", serial, "device", "proximity", flag], adb_path=adb)
            try:
                proximity = json.loads(_run(meta, ["-d", serial, "device", "proximity", "--status", "--json"], adb_path=adb))
            except ValueError as error:
                raise QuestDeviceError("The wear-sensor change could not be verified.") from error
            if proximity.get("enabled") is not (action == "wear-sensor"):
                raise QuestDeviceError("The Quest did not apply the requested wear-sensor setting.")
        if action in {"wake", "keep-awake"}:
            # WAKEUP is idempotent. POWER is a toggle and could put an awake
            # headset back to sleep, so it must never be used here.
            _run(adb, ["-s", serial, "shell", "input", "keyevent", "KEYCODE_WAKEUP"], adb_path=adb)
        state = _state(adb, serial)
        if action in {"wake", "keep-awake"}:
            for _ in range(4):
                if state["wakefulness"] == "Awake" and state["display_on"]:
                    break
                time.sleep(.3)
                state = _state(adb, serial)
            if state["wakefulness"] != "Awake" or not state["display_on"]:
                raise QuestDeviceError("The wake command was sent, but the display is still off. Check the headset power button.")
        messages = {
            "wake": "Quest is awake and its display is on.",
            "keep-awake": "Testing wear override is on and the display is awake. Use Wake headset after a power-button sleep.",
            "wear-sensor": "Normal wear detection is restored. The Quest can detect when you put it on or take it off.",
            "status": f"Quest power state: {state['wakefulness'] or 'unknown'}.",
        }
        return {"action": action, "message": messages[action], **state}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=sorted(ACTIONS))
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(perform_device_action(args.data_dir, args.action), indent=2))
    except QuestDeviceError as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
