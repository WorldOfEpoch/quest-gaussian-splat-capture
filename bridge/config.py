from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    data_dir: Path
    token: str


def load_config(data_dir: Path | None = None, token: str | None = None) -> Config:
    default_parent = Path(os.environ["LOCALAPPDATA"]) if os.name == "nt" and os.environ.get("LOCALAPPDATA") else Path.home()
    root = (data_dir or Path(os.environ.get("QUEST_SPLAT_DATA_DIR", default_parent / "QuestSplatBridge"))).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    token_path = root / "bridge.token"
    configured = token or os.environ.get("QUEST_SPLAT_TOKEN")
    if configured:
        if len(configured) < 32:
            raise ValueError("QUEST_SPLAT_TOKEN must contain at least 32 characters")
        return Config(root, configured)
    if not token_path.exists():
        try:
            with token_path.open("x", encoding="ascii") as handle:
                os.chmod(token_path, 0o600)
                handle.write(secrets.token_urlsafe(48))
        except FileExistsError:
            pass
    saved = token_path.read_text(encoding="ascii").strip()
    if len(saved) < 32:
        raise ValueError("Saved bridge token is invalid")
    return Config(root, saved)
