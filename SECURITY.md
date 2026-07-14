# Security Policy

## Supported versions

The `main` branch is the only supported development line until the project starts publishing tagged releases.

## Reporting a vulnerability

Please do not open a public issue for security-sensitive problems.

Use one of these paths:

- Open a private GitHub security advisory after the repository is published.
- Contact the repository owner privately through the GitHub profile if advisory reporting is not yet enabled.

Include:

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

Maintainers should acknowledge reports as soon as possible, reproduce the issue, patch privately when needed, and publish a security advisory or release notes after a fix is available.
