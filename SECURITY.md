# Security policy

ShelfSense is experimental. There are no supported production releases or promised
security response times; fixes target the current default branch.

## Reporting a vulnerability

Use GitHub's **Security → Report a vulnerability** for this repository when enabled:
https://github.com/Habib-Rehmn/ShelfSense/security/advisories/new

If private reporting is unavailable, open a public issue asking the maintainer to
enable a private reporting channel, without including the vulnerability details,
credentials, exploit payloads, customer footage or personal data. Do not publish
sensitive details in a normal issue or pull request.

Include affected versions/commits, a minimal reproduction with synthetic inputs,
impact and proposed mitigation in the private report. Redact secrets.

## Intended deployment boundary

The API/dashboard have no application authentication and are intended for a trusted
local environment. Compose binds their host ports to localhost. Add authentication,
access controls and transport security before exposing them to other networks.

Treat checkpoints, Torch Hub repositories, videos and dataset files as inputs from
trusted sources. Model loading can execute third-party code; do not accept arbitrary
uploads or weight paths from unauthenticated users. Use the pinned CPU image and
verified downloads documented in `docs/releasing.md`.

Use synthetic data in reports. Dataset consent, store-camera permissions and
retention policies belong to the deployment owner. Do not infer a person's identity
or other sensitive attributes from the temporary tracking IDs.
