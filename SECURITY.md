# Security reporting

This repository contains a source developer preview. The current preview receives source fixes; no public APK or packaged Windows release is supported yet. Future binaries must identify supported versions and security-relevant changes.

Keep the dashboard on loopback. Remote API connections require HTTPS and bearer authentication; pairing tokens, certificate private keys and capture stores belong outside Git. The Blender client refuses HTTP redirects so authenticated requests cannot silently move to another destination. Do not weaken these boundaries to make a connection work.

For a sensitive vulnerability, use GitHub's **Security > Report a vulnerability** private reporting flow if enabled. If that option is unavailable, open an issue asking for a private reporting channel without including the exploit, credentials, or personal data. Do not publish a live pairing token, signing material, private scan, or full device log.

Useful reports name the affected public version, component, expected boundary, reproduction with synthetic data, and potential impact. Report compromised accounts through the relevant platform's normal recovery process; maintainers do not need your password.

Untrusted APKs, mismatched signatures, and certificate failures must not be worked around by disabling checks. Preserve originals and stop the affected workflow while investigating. Sideloading instructions never authorize bypassing Meta entitlement or other account access requirements.
