# Run and test the Windows source

This developer preview provides a source entry point for the PC bridge. It is not an installer for the complete capture/reconstruction system. Python 3.11 and Node.js 22 are the CI baseline; use a source checkout because the Blender launcher and preview simplifier also use sibling repository files.

## Install in a local environment

Follow the commands in [the README](../README.md). They create a repository-local Python environment and install the locked preview simplifier. The `numerical-test` extra installs CPU libraries used by synthetic tests; it does not install CUDA, Nerfstudio, COLMAP, Meta tools, or Blender. `--ignore-scripts` avoids dependency installation hooks; the CPU preview path is tested with this installation.

No environment activation or PowerShell execution-policy change is necessary. Run the executables inside `.venv\Scripts` directly. For Linux contributors, use `python3.11 -m venv .venv` and the equivalent `.venv/bin/python` / `.venv/bin/quest-splat-bridge` paths; CI covers the source tests there, while device setup is maintained for Windows.

## Start, stop and preserve your data

```powershell
.\.venv\Scripts\quest-splat-bridge.exe --data-dir "$env:LOCALAPPDATA\QuestSplatPublic"
```

The dashboard is available at `http://127.0.0.1:8765/ui`. It can create projects and inspect stored results. This first launch has no captures. The server stays in the foreground; Ctrl+C shuts it down. The dashboard's job scheduler can process explicitly requested work or enabled session policies while the server runs. Do not enable automatic processing until its reconstruction environment is configured.

The data folder holds the database, original captures, model revisions, job evidence, and `bridge.token`. Keep it outside Git. A random token is generated on first use unless you deliberately supply `QUEST_SPLAT_TOKEN`; treat it like a password. A data-folder backup is separate from a source checkout. Stop processing before a consistent backup and never delete a store to solve a setup failure.

The dashboard is deliberately enabled only on the plain HTTP loopback listener at port 8765. Changing the port or enabling TLS does not create an alternate dashboard. If that port is already in use, identify the running application; do not kill arbitrary Python processes. The CLI's `--help` lists its API and reconstruction options.

## Run the contributor tests

From the repository root after installing the README dependencies:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -ra
npm test
node scripts/audit-publication.mjs
```

Tests use synthetic inputs and temporary stores. The normal suite does not connect to a headset or train a real scan. Keep `QUEST_NERFSTUDIO_PYTHON` unset unless you intentionally want the optional trainer compatibility check. Optional torch/gsplat/mesh checks are skipped when their dependencies are absent; the output reports each skip. A green CPU suite is not a GPU reconstruction or headset test.

GitHub Actions runs the same CPU suite on Windows and Linux with Python 3.11 and Node 22. The publication check requires an explicit file inventory and rejects common credential/path patterns. It is a guard against accidental inclusion, not a complete secret detector. When adding an intended source file, add its reviewed relative path to `PUBLICATION_FILES.json` too.

## Add reconstruction tools deliberately

The original reconstruction owner invokes `colmap`, `ffmpeg`, `ns-process-data`, `ns-train`, and `ns-export`. They must be installed and discoverable by the chosen environment. GPU training additionally needs a compatible NVIDIA/CUDA, PyTorch, gsplat and Nerfstudio combination. There is no validated one-command installer for that complete stack in this preview.

The supplied `scripts/Build-QuestSplat.ps1` preserves the prototype's tool layout under your chosen data folder: `tools/colmap-4.2.0/bin` and `tools/nerfstudio310-venv/Scripts`. It checks those tools, uses the existing COLMAP compatibility wrapper, and invokes the same bridge CLI. It does not download or provision them. The trainer environment is separate from the Python 3.11 bridge environment. Read [the dependency inventory](public-dependencies.md) before packaging or redistributing any tool.

Optional mesh features use the `mesh` Python extra and its separate licenses. The saved-checkpoint comparison script in `scripts/Compare-QuestSplat.py` requires the GPU environment and writes local comparison images/JSON; those outputs may contain private scenes and paths and are not public fixtures.

## Blender and headset connections

The source includes the Blender add-on and launch helper. The existing Gaussian import path expects Blender 5.3 or newer with the required Gaussian operators. Set `QUEST_SPLAT_BLENDER_EXE` to your chosen executable or make Blender discoverable on PATH. Source availability does not certify a particular experimental Blender build. The add-on's pure Python tests do not launch Blender or prove its viewport behavior.

The localhost dashboard is not a headset endpoint. The Quest client requires HTTPS, bearer-token pairing, project/session/origin identities, and either normal certificate validation or an explicitly paired certificate fingerprint. A non-loopback bridge listener refuses to start without a TLS certificate and key. Do not send a pairing token over plain LAN HTTP or disable certificate validation.

Build preparation is described in [Quest source build](quest-source-build.md). A portable pairing wizard and a freshly validated capture-to-PC setup remain first-release work; no public APK or general-user pairing installer is provided yet. Until those checks exist, use this release for source contribution and isolated development rather than replacing a working capture installation.