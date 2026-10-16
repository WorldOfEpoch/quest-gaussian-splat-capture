# Quest Gaussian Splat Capture

An open project for capturing real scenes with a Meta Quest headset, sending the photographs and tracking information to a Windows PC, and building viewable Gaussian splats and meshes.

**This repository currently publishes documentation and a release plan. There is no public APK, PC bridge download, or application source release here yet.** The existing development prototype has not been packaged for general installation. A source archive from this repository contains these documents, not a working app.

## Sideloading and maintained builds

[Read the Windows sideloading guide](docs/windows-sideloading.md) to prepare your headset and learn how to verify, install, and update a published APK safely. The installation steps apply once a reviewed APK is attached to a project release.

WorldOfEpoch plans to maintain downloadable app builds from the latest reviewed public source tags. Each build will identify its source commit, dependencies, signing certificate, matching PC bridge, and actual headset test results. A new upstream compile is a candidate for testing, not automatically the recommended download. There is no automatic update service or promised release schedule.

Use this repository's [Releases](https://github.com/WorldOfEpoch/quest-gaussian-splat-capture/releases) page when builds become available. Read [the build and update policy](docs/releases-and-builds.md) before switching between community, development, or maintainer builds.

## What we are working toward

1. Install a verified headset APK and its matching Windows bridge with clear setup instructions.
2. Pair the headset with your own PC and capture overlapping views of a scene.
3. Keep the original photographs and calibration local while the PC reconstructs the scene.
4. Inspect the result on the PC or headset, with a clear distinction between an inspection preview and a scan aligned to the physical room.
5. Improve capture, registration, reconstruction, and performance through reproducible community reports.

The headset is the capture and viewing client. **A Windows PC bridge and a compatible reconstruction environment are required for the planned workflow.** Installing an APK alone does not supply the PC software, Meta account access, pairing, or reconstruction tools. Hosted compute and other headsets would require separate implementations and tests.

## Current scope

The prototype targets Quest 3 and the Quest 3S camera API family. Historical physical testing is recorded for Quest 3; Quest 3S is not independently certified by this repository. This is not a Vive client or a phone app. See [project status](docs/project-status.md) for the evidence and remaining release work.

Contributions are welcome: clearer setup instructions, device reports, small reproducible failures, and proposed fixes with before/after evidence. Start with [CONTRIBUTING.md](CONTRIBUTING.md). Avoid posting photographs of private spaces, device identifiers, pairing files, account details, or signing material.

## Documentation

- [Windows sideloading and troubleshooting](docs/windows-sideloading.md)
- [Maintained builds and update compatibility](docs/releases-and-builds.md)
- [Project status and release milestones](docs/project-status.md)
- [Release evidence template](docs/release-evidence-template.md)
- [Security reporting](SECURITY.md)
- [Licensing and dependencies](NOTICE.md)

The original documents in this repository are MIT licensed. No third-party SDK, application source, scan dataset, or binary is included. This community project is independent of Meta, Google, and Unity.
