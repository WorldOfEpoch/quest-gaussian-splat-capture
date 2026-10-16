# Dependencies, notices, and binary release status

Reviewed 2026-10-16. The application's original source is published under the repository license. Dependencies keep their own licenses; installing the development stack does not make the whole stack MIT licensed. No Unity Editor, Meta SDK archive, Python environment, CUDA toolkit, capture dataset, or private app registration is included.

## Browser viewer: included and usable without a CDN

The browser imports two local ES modules. Both modules and their original MIT notices are included in `bridge/static/vendor`; omitting them would leave the viewer unable to load. Together with the notices they occupy **1,912,460 bytes**.

| Included file | Baseline and changes | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `three.module.js` | Three.js 0.160.1 | 1,272,972 | `76dea8151bc9352aef3528b4262e249b2604f62543828328db978d060d61a495` |
| `gaussian-splats-3d.module.js` | GaussianSplats3D 0.4.7, modified locally as described below | 637,329 | `09629bc304733f453a00ac0d1830093232f27e8362d4847e7e1447208f38a5e6` |
| `THREE-LICENSE.txt` | Original Three.js copyright and MIT notice | 1,081 | `852e0e8699169bf9f6fdc6bda3e682d078dcbc738b5d33e74df594721bff271d` |
| `GAUSSIAN-SPLATS-3D-LICENSE.txt` | Original Mark Kellogg copyright and MIT notice | 1,078 | `eaf88512a83fc007481fee3c10b6b6e3dbab64bd6d0d3c64416decf86dbc9440` |

The Gaussian module is a modified distribution, not a pristine upstream bundle. Its two `three` imports use `./three.module.js`. The local SH3 extension parses, packs, transfers, and evaluates all 45 higher-order spherical-harmonic coefficients, including half-precision packing. It preserves the source model's orientation; nonidentity scene rotations are rejected until coefficient rotation is supported. Preserve these changes and the SH3 regression coverage when updating the viewer. Downloading stock 0.4.7 as a replacement would lose that implementation.

The original copyright and permission notices remain required when distributing these modules. Upstream sources: [Three.js license](https://github.com/mrdoob/three.js/blob/r160/LICENSE) and [GaussianSplats3D license](https://github.com/mkkellogg/GaussianSplats3D/blob/main/LICENSE). A future upstream-source build recipe should reproduce this modified bundle before replacing it with generated output.

## Quest build: obtain dependencies separately

The [Quest source build guide](quest-source-build.md) and `quest/dependencies.json` own preparation. This table describes the inspected source baseline, not a newly verified public build.

| Dependency | Recorded input | Terms and packaging |
| --- | --- | --- |
| Unity Editor and Android Build Support | 6000.0.66f2, revision `b20bc5da3050`; Android SDK/NDK/OpenJDK for that Editor | Install separately under [Unity's Editor terms](https://unity.com/legal/editor-terms-of-service/software). |
| Meta Passthrough Camera API sample | Commit `9105be64da8690b41154baf5629cb82dc2dbe4a7` | Fetch into the separate generated workspace. The [sample license](https://github.com/oculus-samples/Unity-PassthroughCameraApiSamples/blob/main/LICENSE.txt) applies by default; only explicitly marked files use other terms. |
| Meta MRUK/Core and Platform SDK | 85.0.0; the Platform SDK archive is a separately supplied, hash-checked input | Meta SDK terms apply. The SDK archive and cached SDK source are not bundled in this repository. |
| UnitySplats | Package 1.2.0, commit `6c0258189a2b124af1282fa9236fd9b6637f1a1a` | MIT; fetch the complete pinned package and preserve its [license and third-party notices](https://github.com/arloopa/UnitySplats). |
| Unity.WebP / libwebp | `com.netpyoung.webp` 0.3.22 | The wrapper uses MIT; libwebp uses BSD-3-Clause. Retain the package's complete notices, including those accompanying native binaries. |
| Unity glTFast | 6.16.1 | [Apache-2.0](https://github.com/Unity-Technologies/com.unity.cloud.gltfast/blob/main/LICENSE.md); resolved as an external package. |
| Unity OpenXR, XR Management, inference, and built-in modules | OpenXR 1.16.1, XR Management 4.5.4, inference 2.4.1; remaining versions in the manifest/lock | Resolve through the pinned Unity project and retain each package's own terms/notices. These are not covered by the application's MIT license. |

UnitySplats' notices also identify gsplat-unity, PlayCanvas Engine, UnityGaussianSplatting, GPUSorting, Spark, SPZ, and ZstdSharp. Its top-level MIT notice is not a substitute for those notices.

Meta's SDK agreement permits defined application distribution and sample-source uses subject to its terms and notice requirements. It does not turn the SDK into MIT code. Keep the SDK and sample notices intact in generated projects and any eventual application distribution. Contributors supply their own appropriate app registration and package identity; the source retains normal entitlement checks. See the [Meta SDK agreement, sections 1.1–1.3](https://developers.meta.com/vr/licenses/oculussdk/).

## PC bridge and optional processing

Python packages are installed from the dependencies declared in `pyproject.toml`; installed environments and package caches are excluded. The following license classifications were checked against the local package metadata/notices or the linked upstream sources.

| Component | Role / recorded version | License notes |
| --- | --- | --- |
| FastAPI | HTTP API; declared range `>=0.115,<1` | MIT |
| Uvicorn | Server; `>=0.30,<1` | BSD-3-Clause |
| python-multipart | Upload parsing; `>=0.0.9,<1` | Apache-2.0 |
| Pillow | Image handling; `>=10,<13` | MIT-CMU |
| NumPy | Numeric data; `>=2,<3` | BSD-3-Clause plus bundled component notices; the inspected wheel reports additional 0BSD, MIT, Zlib, and CC0 terms. |
| httpx / pytest | Test dependencies | BSD-3-Clause / MIT |
| PlayCanvas splat-transform | Optional Quest preview simplification; 3.9.2 in the tool lockfile | [MIT](https://github.com/playcanvas/splat-transform/blob/main/LICENSE); install from the lockfile rather than copying `node_modules`. |
| PlayCanvas engine | Preview tool peer dependency, explicitly pinned to 2.23.3; WebXR typings 0.5.24 are locked transitively | MIT according to the package metadata; installed separately. Recording this peer dependency makes a clean npm install reproducible across npm versions. |
| COLMAP | Optional camera registration and mesh tooling; original reference profile 4.2.0 | COLMAP itself uses [BSD terms](https://github.com/colmap/colmap/blob/main/doc/license.rst). Its compiled dependencies have separate terms; retain the distribution's notices. |
| Nerfstudio / gsplat | Optional Gaussian training; original Windows reference used gsplat 1.5.3 | [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio/blob/main/LICENSE) and [gsplat](https://github.com/nerfstudio-project/gsplat) use Apache-2.0. Obtain separately. |
| PyTorch / CUDA | Optional GPU backend; original reference used PyTorch 2.4.1 with CUDA 12.4 | Separate runtime and vendor terms apply. That reference is not a claim that every GPU, driver, OS, or newer package combination has been tested. |
| PyMeshLab / plyfile | Optional mesh extra; 2025.7.post1 / 1.1.5 | GPLv3 dependencies: [PyMeshLab](https://github.com/cnr-isti-vclab/PyMeshLab/blob/main/LICENSE), [plyfile](https://github.com/dranjan/python-plyfile/blob/master/COPYING). They are not MIT components and are not vendored here. |
| Open3D / SciPy | Optional mesh extra; 0.19.0 / 1.15.3 | MIT / BSD terms, with their distributed dependency notices. |

A future installer or bundled processing environment must inventory the actual resolved packages and satisfy their licenses, including the GPL components and native libraries. The source publication does not include those binaries or claim they can be redistributed as an MIT-only bundle. Hardware/GPU testing and the optional processing installation remain separate from the bridge's Python tests.

## Why the existing development APK is not attached

The newest inspected prototype APK is associated with the original successful Unity build and its source overlay. It is **not a build of the cleaned public source**. The audit found concrete release blockers:

- It is marked debuggable and uses an Android Debug signing certificate.
- Its Unity player configuration embeds development-machine network addresses, and managed metadata contains local installation paths.
- It uses the original app registration and entitlement path. Access for unrelated users has not been established; publishing that file would not grant it.
- The public source now takes the contributor's app identity as configuration. No new APK has yet been built and tested from that configuration.

The ZIP inventory did not contain separately named pairing files, source photographs, splats, or private-key files. That limited inspection is not an exhaustive binary-secret audit. No device was launched, changed, or used to test a new build during this publication audit.

A public binary should come from a recorded public-source revision and resolved dependency set, with deliberate release signing, no development connection payloads, an appropriate app/account distribution arrangement, fresh bridge pairing, and documented headset capture/preview checks. Keep SDK and dependency notices with that distribution. [Android's signing guide](https://developer.android.com/studio/publish/app-signing) explains the distinction between debug and release signing. These APK tasks do not prevent contributors from using and improving the published source now.
