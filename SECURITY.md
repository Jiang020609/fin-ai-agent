# Security Policy

## Reporting a Vulnerability

Please do not open a public issue for security-sensitive reports.

If you find a vulnerability, contact the maintainers privately through the
repository owner's preferred contact channel, then include:

- A short description of the issue.
- Steps to reproduce.
- Potential impact.
- Any suggested fix or mitigation.

## Sensitive Data

This repository should not contain real API keys, `.env` files, private
datasets, generated vector databases, logs, certificates, or model weights.

If a secret is accidentally committed, revoke it immediately and rotate the
credential before cleaning the git history.
