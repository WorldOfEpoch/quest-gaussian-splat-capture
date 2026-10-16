# Project status

Last reviewed: 2026-10-16. This page distinguishes the existing development prototype from what is available in this public repository.

## Publicly available now

| Item | Status |
| --- | --- |
| Windows sideloading guide | Published documentation; installation awaits a public APK. |
| Maintained-build policy and evidence template | Published documentation; no automated release pipeline. |
| Community installation reports and proposed fixes | Accepted through issues and pull requests. |
| Quest application source | Not included yet. |
| Windows bridge source/installer | Not included yet. |
| Signed APK download | Not available yet. |
| Public source build instructions | Pending source/dependency cleanup and a clean-machine build. |
| Device compatibility certification | No public release has been certified. |

## Existing prototype evidence

A read-only review of the existing project's build owner and implementation ledger found recorded Quest 3 camera capture, PC transfer, Gaussian reconstruction, and headset/PC preview work. The prototype also records unresolved reconstruction quality, whole-scene registration, and some physical-headset acceptance work. These historical development receipts do not certify a public artifact that has not been released.

The current source uses package `com.questsplat.bridge`, a pinned Unity build owner, a local Windows bridge, and separate reconstruction tools. It protects session identity and distinguishes saved photographs from admitted/trained views. Photos saved or a reconstruction process completing do not, by themselves, establish visual quality or room alignment.

The reviewed ledger includes private-channel entitlement problems and later physical capture evidence. We have not performed a fresh entitlement or headset test for this public documentation release. The maintained build must state who can actually run it and verify that on a physical headset. A stale failure must not be presented as current, and a later install must not be presented as proof that account access works for everyone.

Quest 3S is a target in the supported camera API family, not an independently tested app claim. Vive support, phone clients, LifePortal integration, and broader VR plugins are possible related work; none is supplied by this repository today.

## Community milestones

1. **Clean source release:** publish the original app and bridge owners with permitted dependencies, private data excluded, and a public license inventory. Acceptance: a contributor can identify every required component and reproduce a build from a clean checkout.
2. **First installable preview:** publish a signed APK, matching Windows bridge, pairing instructions, checksums, and completed test receipt. Acceptance: a new tester can install, connect, capture one real image, and see its durable PC receipt without maintainer-private files.
3. **Reliable updates:** document signer continuity and migration limits. Acceptance: a compatible update preserves pending captures, pairing, project identities, and saved locations, with recovery tested.
4. **Reconstruction quality:** build a small redistributable test dataset with explicit capture consent and provenance. Acceptance: comparisons include fixed viewpoints, registration error, resource use, and visible failure cases; new results preserve the accepted baseline.
5. **Broader compatibility:** add independent Quest 3S and Windows/GPU reports, then evaluate other clients. Acceptance: each entry names the actual software versions and tests, rather than inheriting another device's result.

The shared goal is a capture-to-reconstruction workflow people can install, understand, improve, and trust with their own data. Start with one reproducible problem and an observable acceptance check.
