# Security Policy

[Documentation](docs/README.md) · [Support](SUPPORT.md) · [Code of Conduct](CODE_OF_CONDUCT.md)

## Supported versions

| Version | Supported |
| --- | :---: |
| `main` | ✅ |
| Untagged historical snapshots | ❌ |

The `main` branch is the only supported development line until the project starts publishing tagged releases.

## Reporting a vulnerability

Please do not open a public issue for security-sensitive problems.

Use one of these private paths:

- Open a [private GitHub security advisory](https://github.com/swaroop-kaza/knowledge_graphs/security/advisories/new).
- Contact the repository owner privately through the [GitHub profile](https://github.com/swaroop-kaza) if advisory reporting is unavailable.

Include enough information to reproduce and assess the report:

- A short description of the issue.
- Reproduction steps or proof of concept.
- Affected files, configs, or workflows.
- Whether secrets, PII, remote code execution, unsafe deserialization, dependency compromise, or data exfiltration are involved.

## Security scope

Security-sensitive areas include:

- LLM provider configuration and prompt inputs.
- Graph database credentials and ingestion adapters.
- External dataset download scripts.
- PDF, HTML, XML, JSON-LD, and archive parsing.
- Dead-letter queues, manifests, logs, and generated reports that may contain source data.
- PII redaction and evidence/provenance metadata.

## Handling expectations

| Stage | Maintainer action |
| --- | --- |
| Intake | Keep the report private and confirm that it was received |
| Triage | Reproduce the issue, determine affected surfaces, and assess exposure |
| Remediation | Prepare and verify a private fix when disclosure could create risk |
| Disclosure | Publish an advisory or release notes after a safe fix is available |

> **Minimize sensitive evidence:** do not include live credentials, unnecessary personal data, proprietary corpora, or full production graph exports in a report. Use the smallest synthetic reproduction that demonstrates the issue whenever possible.
