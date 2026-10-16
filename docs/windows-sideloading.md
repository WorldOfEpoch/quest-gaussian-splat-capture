# Sideload Quest Gaussian Splat Capture from Windows

Sideloading installs an Android APK directly from your PC. **No public project APK is available yet.** Complete the preparation steps if useful, then wait for a release containing an APK, checksums, a compatibility receipt, and a matching Windows bridge. Do not mistake GitHub's automatically generated source ZIP for an installer.

## 1. Prepare the headset and Windows PC

Use a Quest 3 or Quest 3S, a USB data cable, and a Windows PC. Check the particular release's tested headset and Horizon OS versions. Meta currently documents its passthrough camera API for Quest 3/3S with Horizon OS 74 or later; that API minimum does not certify every newer OS or this app. Camera access and passthrough must be available. [Meta camera API requirements](https://developers.meta.com/vr/documentation/unity/unity-pca-overview/)

Follow [Meta's device setup guide](https://developers.meta.com/vr/documentation/native/android/mobile-device-setup/) to join or create a developer team, complete account verification, and enable Developer Mode for your paired headset in the Meta Horizon mobile app. Install Meta's Windows ADB driver as directed there. These are personal account steps; this project does not need your Meta password or verification information.

Download the Windows [Android SDK Platform-Tools](https://developer.android.com/tools/releases/platform-tools) from Google and extract them. Open PowerShell in the extracted `platform-tools` folder. This guide uses `.\adb.exe`, so no system PATH changes are necessary.

Connect the headset with the USB data cable, put it on, and accept the USB debugging prompt for your own trusted PC. Then run:

```powershell
.\adb.exe devices -l
```

Find the headset row with state `device`. `unauthorized` means the headset has not approved the computer. An empty list suggests a cable, port, driver, or Developer Mode problem. Device serial numbers are private; remove them from issue reports. Google's [ADB guide](https://developer.android.com/tools/adb) explains device selection and USB authorization.

For the remaining commands, replace `YOUR_QUEST_SERIAL` with that row's serial number. Explicit selection avoids accidentally installing onto another connected phone or headset.

## 2. Choose and verify a release

Use the [project Releases page](https://github.com/WorldOfEpoch/quest-gaussian-splat-capture/releases). Select a release that lists your headset, OS version, and Windows bridge as tested. Experimental releases must say which steps remain unverified.

Read its release notes and download the APK plus `SHA256SUMS.txt`. In PowerShell, compute the APK hash; substitute its actual downloaded path:

```powershell
Get-FileHash -LiteralPath 'C:\Downloads\QuestSplat.apk' -Algorithm SHA256
```

Compare the entire hash to the same release's checksum entry. A mismatch means stop and download again from the release page. A checksum confirms matching bytes; it does not independently establish who built them.

The release must also publish the package ID, version code, and signing-certificate SHA-256 fingerprint. The existing prototype's package is `com.questsplat.bridge`. Check the release receipt rather than assuming an arbitrary APK with that filename has the right identity.

Developers with Android SDK Build-Tools can check an APK's signature using the installed `apksigner.bat`:

```powershell
apksigner.bat verify --verbose --print-certs 'C:\Downloads\QuestSplat.apk'
```

Compare the signer certificate digest with the release receipt. `apksigner` comes from Build-Tools, not the Platform-Tools ZIP. Its [official reference](https://developer.android.com/tools/apksigner) describes signature verification. Keep signing private keys and passwords private; verifying an APK does not require them.

## 3. Protect existing scans before updating

Finish pending uploads and confirm the originals reached your PC. Follow the release's tested backup/export process for the existing installation. Pairing, pending captures, project profiles, and saved spatial anchors may live in app data; generic Android backups cannot be assumed to preserve them.

Android updates require the same package ID, a compatible signing identity, and an equal or higher version code. This project's maintained releases will use increasing version codes. A fork or development build with a different signer cannot be assumed to update the maintainer build. Uninstalling an app removes its app data. [Android update rules](https://developer.android.com/google/play/app-updates)

**Do not uninstall, clear data, or force a downgrade to work around an update error.** The existing development and private-channel build lines use different signers, and no public migration procedure has been validated. If you already have a prototype installation, wait for explicit guidance for your build line before installing a public build over it.

## 4. Install the APK

Once the release checks pass, leave any active capture session cleanly and run this command from the Platform-Tools folder, replacing the serial and APK path:

```powershell
.\adb.exe -s 'YOUR_QUEST_SERIAL' install -r 'C:\Downloads\QuestSplat.apk'
```

Wait for `Success`. The `-r` option preserves existing app data during a compatible replacement; it is not a backup or a guarantee that an app's own migration is correct. Do not add downgrade or permission-grant flags to make a failed installation succeed. [ADB installation options](https://developer.android.com/tools/adb)

On the headset, open the app library's **Unknown Sources** section and select the installed app. Menu placement can vary by Horizon OS release. [Meta's device testing guide](https://developers.meta.com/vr/documentation/android-apps/enable-developer-mode/)

## 5. Confirm the complete capture path

An installation success proves only installation. Follow the matching release's Windows bridge setup and pairing guide before capturing. Those distributable bridge instructions are a release blocker and are not available in this documentation-only version.

Verify these separately:

1. The installed app version matches the release receipt and opens on the headset.
2. The camera and required spatial-data permissions are granted through the normal headset prompts.
3. The app's Meta identity/account access checks pass. The prototype includes entitlement checks; sideloading does not grant account access or bypass Meta registration requirements. [Meta entitlement documentation](https://developers.meta.com/vr/documentation/unity/ps-entitlement-check/)
4. Your own Windows bridge is running, and pairing identifies the intended destination.
5. One ordinary controller capture creates a real photo and matching metadata, with a durable receipt on the PC.
6. Existing scan locations restore correctly after an update. Saved-anchor access also depends on application registration; an untrusted-package error is not fixed by creating a replacement anchor. [Meta anchor troubleshooting](https://developers.meta.com/vr/documentation/unity/unity-ssa-ts/)
7. Reconstruction and headset viewing are checked separately. A saved-photo counter, a successful upload, or a numerical alignment score alone does not prove a good 3D result.

Use a small, nonprivate test scene first. Report the first failing step, the app release, headset model, OS version, and a short sanitized error. Do not publish a home scan or entire device log to demonstrate an install problem.

## Troubleshooting

| Symptom | Next check |
| --- | --- |
| No device listed | Confirm a data-capable cable, direct USB port, official driver, and Developer Mode. |
| `unauthorized` | Put on the headset and approve the trusted PC's USB debugging prompt. |
| More than one device | Use the intended headset's serial with `-s`; do not guess. |
| `INSTALL_FAILED_UPDATE_INCOMPATIBLE` | Stop. Check package/signing compatibility with the maintainer; preserve app data. |
| `INSTALL_FAILED_VERSION_DOWNGRADE` | Select the correct newer release; do not force a downgrade. |
| Installed but absent from the library | Check Unknown Sources and the installed package/version; consult the OS-specific release notes. |
| Camera or spatial permission denied | Review the app's normal permission settings; do not disable device security. |
| Entitlement/untrusted-package error | Check the release's account-access and app-registration requirements. Installation is not entitlement. |
| App opens but uploads fail | Check the matching bridge, pairing, and connection mode. A capture app alone cannot reconstruct on the headset. |
| Model does not align with the room | Preserve the scan and its saved anchor. Report the release and steps; do not reset the origin as a workaround. |

This guide does not open firewall ports, change headset safety settings, install background services, or enable wireless debugging. Use only the connection and permissions documented for the particular release.
