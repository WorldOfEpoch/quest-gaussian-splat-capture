# Build the Quest capture source on Windows

The public [`quest/Assets`](../quest/Assets) tree contains the original application's first-party C#, shaders, Unity metadata, build owner, and regression fixtures. It is an overlay on pinned public dependencies. The private app registration was removed; the source does not bypass Meta entitlement.

**Status:** source is available for contribution. The cleaned public configuration has not yet completed a fresh Unity/Android build or physical-headset test. This guide records a reproducible preparation path and the remaining checks, rather than claiming a verified public APK.

The retained [C# source receipt](../quest/SOURCE_VALIDATION.json) records zero-error compilation of the Editor and Android runtime paths against the existing pinned Unity/SDK reference assemblies. It also records warnings and exact source hashes. This does not test fresh package resolution, shader compilation, Android packaging, or a headset.

## Dependencies

Install Git for Windows and Unity **6000.0.66f2**, including that Editor's Android Build Support, Android SDK/NDK, and OpenJDK. Obtain Unity under its own license. The existing build owner checks this exact Editor version.

Read [`quest/dependencies.json`](../quest/dependencies.json). The bootstrap fetches the public Meta camera sample and UnitySplats at the listed commits into a separate build workspace. It copies only the sample's Assets, Packages, and ProjectSettings into the generated Unity project, overlays our original source, and installs the reviewed package manifest/lock. Vendor source stays separate from this repository and retains its own terms.

Obtain **Meta XR Platform SDK 85.0.0** from [Meta's official SDK download](https://developers.meta.com/horizon/downloads/package/meta-xr-platform-sdk/) under its license. Supply the `com.meta.xr.sdk.platform` package archive, not a ZIP of an arbitrary repository. The script verifies the recorded SHA-256 before copying it into the local build workspace. A newer/different archive needs a deliberate dependency update; it is not silently accepted.

Other resolved packages include MRUK/Core SDK 85, glTFast 6.16.1, Unity OpenXR 1.16.1, inference 2.4.1, and WebP 0.3.22 through OpenUPM. The manifest and lock identify the full versions. Package availability and compatibility must be confirmed during the fresh build; credentials, SDK binaries, and licensed vendor trees are not bundled here.

## Prepare an isolated project

From this repository in PowerShell, choose a **new** build directory outside the public checkout. Replace the example paths with your own:

```powershell
.\scripts\Prepare-QuestUnity.ps1 `
  -Destination 'D:\QuestSplatBuild' `
  -PlatformSdkArchive 'C:\Downloads\com.meta.xr.sdk.platform-85.0.0.tgz' `
  -WhatIf
```

Remove `-WhatIf` to prepare it. The script checks the SDK digest, clones the two pinned public repositories, preserves the source GUIDs, and writes a preparation receipt. It does not launch Unity, install an APK, create a capture session, alter another project, or configure an account. An existing destination is refused rather than overwritten. A failed partial workspace is retained for inspection; use a different new directory for another attempt.

Contributors can check the preparation script without an SDK download or network access:

```powershell
.\quest\tests\Test-PrepareQuestUnity.ps1
```

This Windows test uses the installed .NET Framework compiler to create a temporary fake Git executable and synthetic dependency fixtures. It checks full argument forwarding, paths with spaces, preserved sample subfolders, source/GUID copying, local package references, digest rejection, existing-directory refusal, and `-WhatIf`. It does not prove that upstream downloads or a clean Unity build work. See the [bootstrap validation receipt](../quest/BOOTSTRAP_VALIDATION.json).

Open `D:\QuestSplatBuild\UnityProject` with the pinned Editor and let packages resolve. Review Meta Project Setup requirements. The existing `QuestSceneBuilder` uses the sample's working camera rig and camera-access component, with OpenXR MultiPass and the built-in rendering pipeline. Do not replace it with another rig just to suppress a setup error.

## Supply your app identity

Create or use **your own** appropriate Meta app registration and matching Android package name. Set the two values in the environment used to launch Unity:

```powershell
$env:QUEST_SPLAT_PACKAGE_ID = Read-Host 'Your registered Android package ID'
$env:QUEST_SPLAT_META_APP_ID = Read-Host 'Your Meta app registration ID'
$env:QUEST_SPLAT_BUILD_CONFIGURATION = 'development'
$env:QUEST_SPLAT_APK = 'D:\QuestSplatBuild\QuestSplat-development.apk'
```

The existing builder validates these inputs and writes the app ID to the SDK's settings in the generated project. The runtime uses those settings for SDK initialization and retains its normal entitlement check. Camera/spatial permissions, app registration, account access, bridge pairing, and saved anchors remain distinct requirements. A successful compile grants none of them.

Your fork is not an update for an existing installation unless the package and signer are compatible. Do not reuse the maintainer's registration, overwrite an installed prototype, uninstall it, or clear its data to test this build. Use a deliberate test identity and preserve any existing scans.

## Compile through the existing build owner

Close the generated project's interactive Editor before starting a batch build. Run the installed pinned Editor, adapting its installation path:

```powershell
& 'C:\Program Files\Unity\Hub\Editor\6000.0.66f2\Editor\Unity.exe' `
  -batchmode -nographics -buildTarget android `
  -projectPath 'D:\QuestSplatBuild\UnityProject' `
  -executeMethod QuestSplatBridge.Editor.QuestBatchBuild.BuildAndroid `
  -quit -logFile 'D:\QuestSplatBuild\unity-build.log'
```

`QuestBatchBuild` builds the scene, runs its existing HUD and metadata fixtures, checks camera/manifest settings, chooses ARM64, and requires a successful Unity BuildReport and a nonempty APK. It preserves the original signing settings after a release build. It does not establish a public release or install to a device.

For a release build, the same owner accepts `QUEST_SPLAT_BUILD_CONFIGURATION=release`, `QUEST_SPLAT_RELEASE_KEYSTORE`, `QUEST_SPLAT_RELEASE_ALIAS`, and `QUEST_SPLAT_RELEASE_PASSWORD`. Provision these privately in the build process; never commit signing files/passwords or copy generated project settings into the public repo. Each contributor uses their own signer. The source retains the prototype's target API 34 for release and 35 for development; review current Meta distribution requirements before a channel/store upload.

Check both the process result and build log, then inspect APK identity, signer, permissions, and contents. A no-error source compilation does not validate shaders, Android packaging, SDK entitlement, or physical behavior. Share a sanitized [release evidence receipt](release-evidence-template.md), not the raw build directory.

## Validate before recommending a build

Use [Windows sideloading](windows-sideloading.md) only after signer/data compatibility is established. Pair with the corresponding Windows bridge and test the ordinary controller capture path: a real image and metadata on the headset, a durable matching receipt on PC, restoration of a saved location, reconstruction, and actual wearer viewing. Read the [current release policy](releases-and-builds.md).

The source includes recovery and cleanup policy, not permission to discard original photographs or reset scan origins. Report the exact failing stage and preserve the previous accepted data and model while testing a candidate.
