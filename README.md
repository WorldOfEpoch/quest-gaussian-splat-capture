# Quest Gaussian Splat Capture

Capture real scenes with a Meta Quest headset, send photographs and tracking information to a Windows PC, and reconstruct viewable Gaussian splats and meshes. An open project maintained by WorldOfEpoch, with community fixes and compatibility reports welcome.

**The application source is available now as a developer preview.** This includes the PC bridge and dashboard, Quest capture source, Blender add-on, reconstruction owners, and automated tests. There is no public APK or Windows installer yet. A source checkout can run the PC dashboard; headset capture also needs a separately built Quest app, your own Meta registration, secure pairing, and reconstruction dependencies.

## Start with the Windows bridge

Install Python 3.11, Git, and Node.js 22 or 24. In PowerShell:

```powershell
git clone https://github.com/WorldOfEpoch/quest-gaussian-splat-capture.git
cd quest-gaussian-splat-capture
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test,numerical-test]"
npm ci --prefix tools/splat-transform --ignore-scripts --no-audit --no-fund
.\.venv\Scripts\quest-splat-bridge.exe --data-dir "$env:LOCALAPPDATA\QuestSplatPublic"
```

Open [the local dashboard](http://127.0.0.1:8765/ui). Keep that terminal open; stop with Ctrl+C. The example uses a separate data folder so it does not open an existing prototype store. If port 8765 is occupied, stop and identify its owner before choosing how to proceed.

Read [Windows source setup](docs/windows-source-setup.md) for tests, data locations, optional reconstruction tools, and the limits of this first preview. This command starts an empty local bridge; it does not install a service or configure a headset.

## Work on the source

| Component | Existing owner | Starting point |
| --- | --- | --- |
| PC API, capture receipts and project storage | `bridge/app.py`, `bridge/store.py` | [Windows setup](docs/windows-source-setup.md) |
| PC dashboard and 3D viewer | `bridge/dashboard.py`, `bridge/static/viewer.js` | `tests/bridge/` and browser module tests |
| Reconstruction, model revisions and preview quality | `bridge/reconstruct.py`, `bridge/online.py`, `bridge/region_edit.py` | [Dependency inventory](docs/public-dependencies.md) |
| Quest capture and viewing | `quest/Assets/Scripts/` | [Quest source build](docs/quest-source-build.md) |
| Unity build and fixtures | `quest/Assets/Editor/` | Pinned dependency preparation and existing build owner |
| Blender integration | `blender_addon/quest_splat_bridge/`, `blender_addon/open_scan.py` | `tests/blender/` |

The published Unity files overlay pinned upstream projects; this checkout is not a self-contained Unity project. The preparation script obtains separate dependencies into a new build workspace. Meta SDKs, Unity, signing material and personal captures are not bundled. Existing entitlement checks remain enforced.

Fork the repository, work on a focused branch, and open a pull request. Start with [CONTRIBUTING.md](CONTRIBUTING.md), [open issues](https://github.com/WorldOfEpoch/quest-gaussian-splat-capture/issues), and [the current milestones](docs/project-status.md). No headset is needed to improve the bridge's synthetic-data tests or documentation.

## The goal

A new user should be able to install a verified headset app and matching Windows bridge, pair with their own PC, capture a scene, recover from a disconnect without losing photos, and inspect a reconstruction with honest quality and alignment indicators. Contributors should be able to reproduce a problem and prove a fix without receiving somebody else's account or private scan.

The prototype targets Quest 3 and the Quest 3S camera API family. Historical physical testing exists for Quest 3; this cleaned source has not received a fresh end-to-end headset test. Quest 3S remains unverified. A Windows PC and a compatible reconstruction environment are required for the full workflow. Vive, phone, LifePortal and other clients are future integration work, not features delivered by this release.

## Sideloading and maintained builds

WorldOfEpoch plans to maintain builds from reviewed public source tags. Each download will state its exact source, dependencies, signer, matching bridge, and actual device test results. There is no automatic updater or promised release schedule.

- [Windows sideloading guide](docs/windows-sideloading.md)
- [Maintained builds and update compatibility](docs/releases-and-builds.md)
- [Source preview evidence and limits](docs/source-preview-evidence.md)
- [Release evidence template](docs/release-evidence-template.md)
- [Releases](https://github.com/WorldOfEpoch/quest-gaussian-splat-capture/releases)

Original source and documentation are MIT licensed. Included browser libraries retain their MIT notices; separately obtained SDKs and tools retain their own licenses. See [NOTICE.md](NOTICE.md) and [the dependency inventory](docs/public-dependencies.md). This community project is independent of Meta, Google, Unity and OpenAI.