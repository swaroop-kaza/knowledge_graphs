# Support

[Documentation](docs/README.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

Choose the channel by outcome:

| Need | Channel | Include |
| --- | --- | --- |
| Bug or reproducible failure | [GitHub issue chooser](https://github.com/swaroop-kaza/knowledge_graphs/issues/new/choose) | Command, profile, traceback, minimal input, and expected behavior |
| Documentation gap | [GitHub issue chooser](https://github.com/swaroop-kaza/knowledge_graphs/issues/new/choose) | Page link, unclear section, and suggested correction |
| Feature request | [GitHub issue chooser](https://github.com/swaroop-kaza/knowledge_graphs/issues/new/choose) | Construction or evaluation use case, data shape, and acceptance criteria |
| Architecture or research discussion | [GitHub issue chooser](https://github.com/swaroop-kaza/knowledge_graphs/issues/new/choose) | Context, constraints, alternatives, and supporting references |
| Security or private-data concern | Private process in [SECURITY.md](SECURITY.md) | Minimal non-public reproduction details |

## Before opening a report

1. Search [existing issues](https://github.com/swaroop-kaza/knowledge_graphs/issues) for the error or capability name.
2. Confirm the setup and profile requirements in [Getting started](docs/getting_started.md).
3. Run the narrowest relevant check from the [testing guide](docs/testing_guide.md).
4. For provider or service failures, include the output of `uv run kg-construction doctor --strict`.

Do not paste credentials, private source data, proprietary graph exports, or unredacted evidence into a public issue. Follow [SECURITY.md](SECURITY.md) when a report cannot be safely disclosed.

Generated research scores are not product certificates. Questions about that distinction should start with the [evaluation guide](docs/evaluation_metrics.md) and [research boundary](research/README.md).
