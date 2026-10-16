# Project status

Last reviewed: 2026-10-16. This page distinguishes the existing development prototype from this public source preview.

## Available in the source preview

| Item | Status |
| --- | --- |
| PC bridge, dashboard, storage and reconstruction source | Included; editable Python setup and CPU regression tests. |
| Quest application source | Included as an original-source overlay with pinned dependency preparation. |
| Blender add-on and launch helper | Included; pure Python tests, no fresh Blender viewport acceptance. |
| Browser 3D viewer | Included with the modified SH3 renderer and upstream MIT notices. |
| Windows sideloading guide and maintained-build policy | Included; no public APK yet. |
| Windows installer / complete GPU environment installer | Not available yet. |
| Clean Unity/Android build of the public app identity configuration | Pending; reference-assembly C# checks are narrower evidence. |
| Public hardware compatibility certification | No installable public release has been certified. |

Read [source-preview evidence](source-preview-evidence.md) for actual test results and skipped checks. Contributors can work on the source now; they do not need the maintainer's private working directory.

## Existing prototype evidence

Earlier development records describe Quest 3 camera capture, PC transfer, Gaussian reconstruction and headset/PC preview. They also record unresolved reconstruction quality, whole-scene registration and physical-headset acceptance work. These historical receipts do not certify the cleaned public configuration.

The prototype used a private app registration. The public builder requires a contributor's own matching package and Meta registration while preserving entitlement checks. A sideload, compile or private-channel launch does not establish account access for everybody.

Photos received, views admitted to training, model revision updates, numerical alignment and physical visual acceptance are distinct states. Preserve original capture/session identities and the prior accepted model while reviewing new candidates.

Quest 3S is a target in the camera API family, not an independently tested app claim. Vive support, phone clients, LifePortal integration and broader VR plugins are related future work; none is supplied as a complete integration by this repository today.

## Community milestones

1. **Source contribution baseline — published.** Original owners, source tests, dependency inventory and setup guides are available. Remaining acceptance: reproduce the complete Unity/Android build on a fresh contributor machine using separately obtained dependencies.
2. **First installable preview.** Publish a signed APK, matching Windows bridge, portable pairing setup, checksums and a completed test receipt. Acceptance: a new tester installs, connects, captures one real image and sees its durable PC receipt without maintainer-private files.
3. **Reliable updates.** Document signer continuity and migration limits. Acceptance: a compatible update preserves pending captures, pairing, project identities and saved locations; recovery is tested.
4. **Reconstruction quality.** Establish a small redistributable dataset with consent and provenance. Acceptance: comparisons use fixed views, registration error, resource use and visible failure cases, preserving the accepted baseline.
5. **Broader compatibility.** Add independent Quest 3S and Windows/GPU reports, then evaluate other clients. Acceptance: every entry names actual versions and tests instead of inheriting another device's result.

The shared goal is a capture-to-reconstruction workflow people can install, understand, improve and trust with their own data. Start with one reproducible problem and an observable acceptance check.