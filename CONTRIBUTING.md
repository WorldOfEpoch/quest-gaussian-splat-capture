# Contributing

Help make Quest capture and Windows reconstruction installable, reproducible, and understandable. The source is available as a developer preview; the first public APK and complete installer are still being prepared.

Fork this repository, create a focused branch, and open a pull request. See [Windows source setup](docs/windows-source-setup.md) for a local bridge and CPU tests, and [Quest source build](docs/quest-source-build.md) for the Unity overlay. The [project milestones](docs/project-status.md) describe useful work and its acceptance checks. A headset is not required for bridge, viewer or documentation contributions.

## Report a problem

Search existing issues first. Add a comment to the matching issue when you have the same failure, stating your versions and whether the documented workaround helped. Open a new issue for a distinct failure.

Include the release/tag, Windows version, headset model, Horizon OS version, exact failing step, expected result, actual result, and a short sanitized error. Distinguish installation, app startup, entitlement, permissions, pairing, capture, transfer, reconstruction, and viewing.

Do not attach whole diagnostic archives. Remove usernames, device serials, home addresses, account identifiers, network addresses, pairing files, tokens, and private paths. Use a small nonprivate scene or synthetic fixture when visual evidence is necessary. Only contribute data you may redistribute.

## Propose a fix

Describe one problem, its likely cause, and the observable outcome. Reproduce the failure when possible, change the existing owner, and show before/after evidence. Preserve capture/session identities and original data. A fix must not silently reset a scan, clear application data, bypass permissions, or weaken certificate/signature checks.

Documentation improvements should name the tool/OS versions checked and link primary platform documentation. Software contributions should include focused regression checks and state whether a physical headset was used. Run the documented Python/browser tests; add new intended files to `PUBLICATION_FILES.json` after checking their contents. Do not claim device acceptance from compilation alone.

Use the [release evidence template](docs/release-evidence-template.md) for build reports. Mark skipped checks as not run. Keep dependency upgrades, behavior changes, and unrelated cleanup separate when it makes review easier.

## Ownership and licensing

Contribute only material you have the right to share under the applicable license. Original code and documentation contributions use this repository's MIT license. Third-party application code, SDKs, assets, datasets, and binaries need their own license review before inclusion. Read [NOTICE.md](NOTICE.md).

This project does not need maintainer signing keys, Meta account credentials, or personal capture stores in a pull request. Report security-sensitive issues using [SECURITY.md](SECURITY.md).
