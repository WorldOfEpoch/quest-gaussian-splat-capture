# Licensing and dependency notice

The repository's MIT license covers WorldOfEpoch's original application source, tools, tests, documentation and contribution templates. This source preview includes the PC bridge, Quest source overlay, Blender integration and CPU test fixtures. It does not relicense any third-party component.

Two browser runtime modules are included under their original MIT licenses in `bridge/static/vendor/`, with the accompanying license files. The Gaussian renderer includes local SH3 support changes; retain both its upstream notice and the recorded modification provenance. Exact versions and digests are in [the dependency inventory](docs/public-dependencies.md).

Unity, Meta SDKs and samples, the Unity Gaussian renderer, Android tools, Python/npm packages and reconstruction tools are obtained separately. Their manifests record versions and origins; their licenses remain applicable. Optional mesh tools include copyleft components. Do not assume that the project's MIT license permits repackaging every dependency into one binary.

No private capture dataset, APK, development environment, signing material, personal device configuration, Meta account registration, or licensed SDK tree is included. Generated build workspaces and private runtime data stay outside the published source tree.

Meta, Quest, Horizon OS, Android, Unity, Blender and other product names identify compatibility targets or dependencies. This project is not an official product of, or endorsed by, those owners.