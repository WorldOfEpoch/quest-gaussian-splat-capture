import json
import re

import pytest
from fastapi.testclient import TestClient

from bridge.app import create_app
from bridge.config import Config
from bridge.quest_device import QuestDeviceError, _select_device, perform_device_action


def fake_device(tmp_path, monkeypatch):
    adb = tmp_path / "adb.exe"
    meta = tmp_path / "metavr.exe"
    adb.touch()
    meta.touch()
    (tmp_path / "device-tools.json").write_text(json.dumps({
        "adb_path": str(adb), "meta_cli_path": str(meta), "device_id": "QUEST_TEST"}))
    calls = []
    sensor = {"enabled": False}
    def run(executable, arguments, *, adb_path):
        assert adb_path == adb
        calls.append((executable, arguments))
        if arguments == ["devices", "-l"]:
            return "List of devices attached\nQUEST_TEST device model:Quest_3\nOTHER device model:Phone\n"
        assert arguments[:2] in (["-s", "QUEST_TEST"], ["-d", "QUEST_TEST"])
        if "proximity" in arguments:
            if "--enable" in arguments:
                sensor["enabled"] = True
            elif "--disable" in arguments:
                sensor["enabled"] = False
            return json.dumps(sensor)
        if arguments[-2:] == ["dumpsys", "power"]:
            return "mWakefulness=Awake\nmLastSleepReason=power_button\n"
        if arguments[-2:] == ["dumpsys", "display"]:
            return "mScreenState=ON\n"
        if arguments[-2:] == ["dumpsys", "battery"]:
            return "level: 85\n"
        return ""
    monkeypatch.setattr("bridge.quest_device._run", run)
    return calls


def test_wake_targets_only_configured_quest_without_power_toggle(tmp_path, monkeypatch):
    calls = fake_device(tmp_path, monkeypatch)
    result = perform_device_action(tmp_path, "wake")
    assert result["display_on"] and result["wakefulness"] == "Awake"
    assert result["battery_percent"] == 85
    mutations = [args for _, args in calls if "keyevent" in args]
    assert mutations == [["-s", "QUEST_TEST", "shell", "input", "keyevent", "KEYCODE_WAKEUP"]]
    assert all("settings" not in args and "proximity" not in args for _, args in calls)


def test_wear_detection_can_be_restored_and_testing_explicitly_wakes(tmp_path, monkeypatch):
    calls = fake_device(tmp_path, monkeypatch)
    perform_device_action(tmp_path, "wear-sensor")
    assert any("--enable" in args for _, args in calls)
    assert all("keyevent" not in args for _, args in calls)
    calls.clear()
    perform_device_action(tmp_path, "keep-awake")
    assert any("--disable" in args for _, args in calls)
    assert any("KEYCODE_WAKEUP" in args for _, args in calls)


def test_unavailable_ambiguous_or_unauthorized_device_is_not_targeted():
    with pytest.raises(QuestDeviceError):
        _select_device("Q1 device model:Quest_3\nQ2 device model:Quest_3\n", None)
    with pytest.raises(QuestDeviceError):
        _select_device("Q1 unauthorized\n", "Q1")
    with pytest.raises(QuestDeviceError):
        _select_device("PHONE device model:Phone\n", "QUEST")
    assert _select_device("Q1 device model:Quest_3\nPHONE device model:Phone\n", None) == "Q1"


def test_failed_wake_is_reported_not_claimed_as_success(tmp_path, monkeypatch):
    fake_device(tmp_path, monkeypatch)
    monkeypatch.setattr("bridge.quest_device._state", lambda *args: {
        "wakefulness": "Asleep", "display_on": False, "battery_percent": 85})
    monkeypatch.setattr("bridge.quest_device.time.sleep", lambda seconds: None)
    with pytest.raises(QuestDeviceError, match="display is still off"):
        perform_device_action(tmp_path, "wake")


def test_dashboard_device_controls_require_local_csrf_and_fixed_action(tmp_path, monkeypatch):
    app = create_app(Config(tmp_path, "t" * 48), dashboard_enabled=True)
    local = TestClient(app, base_url="http://127.0.0.1:8765", client=("127.0.0.1", 50000))
    page = local.get("/ui")
    assert "Wake headset" in page.text and "Restore wear detection" in page.text
    token = re.search(r"name='csrf' value='([^']+)'", page.text).group(1)
    calls = []
    monkeypatch.setattr("bridge.app.perform_device_action", lambda root, action:
                        calls.append(action) or {"message": "Quest is awake and its display is on."})
    assert local.post("/ui/headset/wake", data={"csrf": "wrong"}).status_code == 403
    assert local.post("/ui/headset/reboot", data={"csrf": token}).status_code == 422
    assert calls == []
    response = local.post("/ui/headset/wake", data={"csrf": token})
    assert response.status_code == 200 and "Quest is awake and its display is on." in response.text
    assert calls == ["wake"]
    lan = TestClient(app, base_url="http://192.168.0.136:8765", client=("192.168.0.38", 50000))
    assert lan.post("/ui/headset/wake", data={"csrf": token}).status_code == 403
    assert calls == ["wake"]
    assert "&lt;script&gt;" in local.get("/ui?headset_notice=%3Cscript%3E").text
