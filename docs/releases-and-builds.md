# Maintained builds and update policy

WorldOfEpoch intends to maintain reviewed Quest APKs and a compatible Windows PC bridge built from public, tagged source. This repository is the planned release home. **No APK or bridge package has been published here yet, and application source is not yet available for a public build.**

## What “updated public builds” means

The maintainer will select a public source revision, review dependency changes, compile the app, and publish it after the relevant checks. The release will link its exact source tag and commit. “Latest” means the newest published release with its stated evidence; it does not mean every upstream commit, SDK update, or local experiment has been tested.

Preview releases may be useful for focused testing when their limits are explicit. A preview must still pass the privacy, license, package-identity, and signature checks below. A recommended release additionally needs successful normal headset capture, PC receipt, preserved update data, and the advertised viewing behavior on the named hardware. Untested hardware remains unverified.

There is no scheduled build service, automatic updater, or guaranteed delivery interval today. Use GitHub's repository watch settings for release notifications if desired.

## Before the first software release

- Extract the original application owners into a clean public source tree. Keep personal capture stores, account settings, logs, machine-specific configuration, and signing material out.
- Audit third-party licenses and redistribution requirements. Dependency presence in a local prototype does not authorize republishing its source or binaries.
- Provide a reproducible dependency manifest and one documented Windows build path, including pinned Unity/Android versions and the PC reconstruction environment.
- Establish the public signing and account-access plan. Preserve existing installations; do not silently replace an app identity or require destructive migration.
- Package and test the Windows bridge setup, pairing, first capture, disconnect/reconnect, and explicit shutdown. A fresh installation must not depend on the maintainer's private files.
- Complete a review of camera, spatial data, network access, local storage, and photo export/deletion behavior.

The existing prototype's build owner pins Unity `6000.0.66f2`, ARM64 Android, and package `com.questsplat.bridge`. It has separate development and release modes and loads release signing inputs outside source. These are inspected prototype facts, not instructions to build this documentation-only checkout or a promise about future minimum versions.

## Required release evidence

Every software release should include the [completed evidence template](release-evidence-template.md), with:

- Public source repository, tag, full commit, and build recipe.
- APK filename, byte size, SHA-256, package ID, version name/code, signing-certificate SHA-256, and build mode.
- Matching Windows bridge version and protocol, dependencies, GPU requirements, and tested Windows version.
- Headset model and Horizon OS version for each actual test, with date and result.
- Fresh-install and compatible-update results, including pending uploads, pairing, project profiles, and saved anchors.
- Meta account/channel requirements where applicable. A downloadable file must not be advertised as generally usable when entitlement is still restricted.
- Known failures, unsupported features, migration instructions, and rollback limitations.

Attach checksums and notices alongside the binaries. Record failures honestly. A compile, upload, install, launch, controller capture, PC receipt, reconstruction, and worn-headset review are separate states.

## Signing and forks

Retain one documented signing identity for each maintained update line. Publish its certificate fingerprint, never its private key. Any deliberate signing transition needs a validated migration and data-preservation plan. Android signing and update compatibility are described in [Sign your app](https://developer.android.com/studio/publish/app-signing) and [How app updates work](https://developer.android.com/google/play/app-updates).

Community builds must identify their fork, source commit, signer, and tests. They must not impersonate a maintainer release or silently replace it. If a fork uses a separate package identity, it also needs its own appropriate Meta registration and a documented data-transfer path; changing a package name alone does not preserve saved anchors or grant platform access.

Do not upload a local APK merely because it compiled. The private prototype's development signer and channel signer differ, so those existing binaries are not interchangeable update packages. No blanket migration is approved by this repository.

## Upstream changes and community improvements

Update dependencies in a focused change, keep the prior version recorded, and reproduce the affected behavior. A new OS or SDK can change camera, anchor, permission, rendering, or signing behavior. Reports should name exact versions and distinguish source tests from physical-device evidence.

Contribute build fixes and test results through pull requests. Release signing remains with the maintainer; contributors should be able to build and test their own clearly identified development artifacts without receiving maintainer credentials.
