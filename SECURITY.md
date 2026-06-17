# Security Policy

## Supported versions

Security fixes are handled for the latest released major version of Bumblebee.

| Version | Supported |
| --- | --- |
| 2.x | Yes |
| 1.x | No |

## Reporting a vulnerability

Please do not open a public GitHub issue for security reports.

Use GitHub private vulnerability reporting if it is enabled for the repository:

```text
https://github.com/arpan404/bumblebee/security/advisories/new
```

If private reporting is not available, contact the maintainer privately through GitHub.

## Scope

Security reports may include:

- unsafe default behavior in mouse or keyboard automation
- unexpected execution of real clicks or typing
- packaging issues that could ship unintended files
- dependency vulnerabilities that affect normal Bumblebee usage

Training datasets, local checkpoints, and user-created automation scripts are outside the package security scope unless Bumblebee itself causes unsafe behavior.
