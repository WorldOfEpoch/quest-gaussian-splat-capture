# Quest capture application source

This is the first-party Unity source overlay from the working Quest Splat Bridge prototype: capture and upload, retained session identity, anchor restoration, project menus, Gaussian/mesh viewing, controller HUD, cleanup policy, shaders, scene authoring, and regression fixtures. Existing Unity GUIDs are preserved.

It is an overlay, not a complete vendor-inclusive Unity project. [Prepare and build the isolated project](../docs/quest-source-build.md) using the pinned public dependencies. The bootstrap creates a separate build workspace; it does not touch an installed app, headset, capture store, or existing Unity project.

The existing controller and build owner are retained. The public version reads the Meta app ID from the SDK's generated settings instead of shipping a maintainer's registration. The builder requires your `QUEST_SPLAT_META_APP_ID` and `QUEST_SPLAT_PACKAGE_ID`. Runtime entitlement checks still run; source availability does not grant access to another developer's Meta app or anchors.

The shared request-configuration helper disables redirects for authenticated bridge traffic, including session/project and preview requests. This change has source-compilation evidence; redirect behavior has not been retested on a headset.

The source depends on Meta's camera sample, XR/Platform SDKs, UnitySplats, glTFast, and Unity packages under their respective licenses. They are fetched or supplied separately and are not relicensed by this repository. See [dependency information](dependencies.json) and [the project notice](../NOTICE.md).

The original private prototype was built and tested previously. This cleaned source has not yet produced a freshly validated public APK. Source checks, full Unity compilation, APK signing, installation, entitlement, normal capture, PC receipt, and wearer acceptance must be reported separately.
