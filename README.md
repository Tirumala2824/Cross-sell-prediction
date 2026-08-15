# Cross-sell Prediction

> A reproducible machine-learning project for exploring customer cross-sell classification with multiple baseline models and evaluation metrics.

[![CI](https://github.com/Tirumala2824/Cross-sell-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Tirumala2824/Cross-sell-prediction/actions/workflows/ci.yml)

## Status

**Category:** Research or ML.

**Lifecycle:** Public experiment in the transition from script-based analysis to reproducible model evaluation. It is not a production decisioning service.

## Project scope

The repository contains training and test data, logistic-regression and support-vector-machine experiments, ROC/AUC evaluation code, and supporting scripts. The intended engineering outcome is a transparent baseline comparison rather than a deployable underwriting or sales-automation system.

## Reproducibility

The data contract and repository structure are protected by tests. Before running the experiments, create an isolated Python environment, install the project’s required scientific dependencies, and inspect the source scripts for the expected columns and file paths. Future improvements should add a pinned dependency manifest, explicit preprocessing steps, deterministic seeds, model artifacts with provenance, and a single documented evaluation command.

## Architecture

```text
train.csv / test.csv
    -> preprocessing and feature preparation
    -> model baselines
    -> ROC/AUC and classification metrics
    -> documented experiment results
```

Keep data loading, transformation, training, and evaluation separate so that the same preprocessing is used for comparison and later inference. See [`docs/engineering-standards.md`](docs/engineering-standards.md).

## Testing and quality

```bash
pytest -q
```

CI validates repository structure and data contracts. Model quality claims should include dataset provenance, split methodology, class-balance treatment, metrics, and known limitations.

## Responsible use

The repository is an educational and research artifact. Do not use its output for consequential decisions without domain validation, bias analysis, privacy review, monitoring, and an approved deployment architecture.

## Contributing and license

See [`CONTRIBUTING.md`](CONTRIBUTING.md), [`SECURITY.md`](SECURITY.md), and [`CHANGELOG.md`](CHANGELOG.md). The repository is released under the MIT License; see [`LICENSE`](LICENSE).
