# Security

Please report vulnerabilities privately through GitHub's "Report a vulnerability" (Security →
Advisories) on this repository, not in a public issue. You'll get an acknowledgement within a
week.

Deployment notes: the server binds to 127.0.0.1 by default. If you expose it, set API keys
(`--api-key` or `TD_API_KEYS`) and consider `--rate-limit-rpm`.
