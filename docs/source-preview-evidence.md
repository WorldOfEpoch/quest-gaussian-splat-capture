# Source preview evidence

Reviewed 2026-10-16 for `v0.1.0-dev.1` (Python package version `0.1.0.dev1`). This is a source contribution release, not an installable headset release.

## Published scope

The public checkout contains the original PC bridge, dashboard/viewer, reconstruction and revision owners, Blender add-on/launch helper, Quest C#/shader source overlay, dependency manifests, contribution guides and synthetic tests. Only the two reviewed MIT browser modules are vendored, with their notices and hashes.

The extraction excludes personal capture stores, model artifacts, operational logs, environments, APKs, device configuration, app registration values, signing material, and Meta SDK/sample source. The source preparation script obtains pinned upstream dependencies into an isolated build workspace. `PUBLICATION_FILES.json` records the exact intended source files; the publication guard checks the checkout and the release archive before upload.

## Changes made for public use

- The Quest build owner requires contributor-supplied Android package and Meta app identity. SDK initialization reads that configuration; entitlement enforcement remains.
- Authenticated Quest requests share a redirect refusal in the existing certificate-configuration helper. This source change has compilation evidence, not a new physical-device receipt.
- The Blender client refuses redirects for both JSON requests and model uploads. Fifteen real loopback fixture cases cover GET, JSON POST and multipart POST against 301, 302, 303, 307 and 308 responses, asserting that a second origin receives no request or pairing token.
- The Blender launcher discovers an explicitly configured executable or PATH entry instead of a maintainer-specific installation path.
- Runtime probe tests use explicit synthetic tokens even when a developer has a real pairing token in their environment.
- The Unity preparation script uses explicit native argument arrays and one selected Git executable. Its offline fixtures cover paths with spaces, preserving upstream Assets, source/GUID copying, package URI resolution, dependency hashes, destination refusal and a no-write preview.

## Checks and their limits

| Check | Result |
| --- | --- |
| Editable Python installation in a new Python 3.11 environment | Passed; dependency consistency check passed. |
| Python sdist/wheel build | Passed; browser runtime assets are included. The source checkout remains required for sibling Blender/tools files, so no standalone wheel is advertised as the complete app. |
| Python bridge and Blender suite on Windows | **539 passed, 38 skipped**. Synthetic temporary stores; no headset or live capture store used. |
| Browser module tests | **14 passed**: viewer navigation, region-review actions and SH3 coefficient preservation. |
| Unity preparation fixture on Windows | **14 checks passed** using synthetic Git/SDK/sample inputs, without fetching or launching Unity. |
| Quest Editor and Android C# reference-assembly compilation | Passed with zero errors; existing obsolete API/unused/unassigned-field warnings are retained in `quest/SOURCE_VALIDATION.json`. |
| Public dependency and privacy inventory | Reviewed; common credential/path guard passed. This is not a complete security certification. |
| GitHub Windows/Linux CPU compatibility | Workflow included. For each release, verify the run for its exact source commit in [Actions](https://github.com/WorldOfEpoch/quest-gaussian-splat-capture/actions). |

The 38 skipped Python checks require optional mesh/texture libraries, PyTorch/gsplat, or an explicitly selected Nerfstudio environment. One test-dependency deprecation warning concerns Starlette's current httpx TestClient integration. No failures were hidden or converted into passes. The preview simplification checks use the actual locked CPU tool with synthetic Gaussian data.

The C# check used the pinned Editor's compiler and existing dependency reference assemblies. It does **not** establish fresh dependency resolution, Unity project import, shader compilation, Android packaging, signing, entitlement, camera capture, PC transfer, room alignment, reconstruction quality or wearer acceptance. No GPU training or Blender viewport check was run for this source publication.

## Why there is no APK attached

The existing development binary is not a build of this cleaned source. Its audited contents include development connection settings and local installation paths; it uses debug signing and the previous private entitlement configuration. Publishing it would not give unrelated users a clean, generally runnable release.

The next release target is a fresh public-source build with a deliberate signing/account-access plan, matching bridge setup and pairing, sanitized binary inspection, and an actual headset capture/receipt/viewing test. See [project milestones](project-status.md), [dependency details](public-dependencies.md) and [the build policy](releases-and-builds.md).
