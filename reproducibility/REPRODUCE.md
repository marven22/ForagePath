# Reproducing the ForagePath paper artifacts

## Scope

This package supports two levels of reproduction.

1. **Artifact verification** checks the compact frozen metrics, strict evidence ledger, candidate lists, and cohort definitions used in the manuscript. It does not require restricted raw records.
2. **End-to-end recomputation** rebuilds the data products and reruns models from the original public sources. It requires substantial download time, source-specific terms compliance, and, for a small number of services, locally obtained credentials.

The source code used for the manuscript's flowering, bee-timing, link-ranking, mixture-replay, uncertainty, and sensitivity analyses is retained in `scripts/`. The paper experiment registry records which artifacts support each reported claim.

## Environment

The archived analysis was executed with Python 3.13 and the packages pinned in `requirements-research.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-research.txt
```

## Verify the release

```powershell
python scripts/verify_paper_release.py
```

The verifier checks the published compact outputs against the paper-release manifest. It validates the 1,138 candidate plants, 436 native-bee targets, 5,236 Grade A/B evidence links, the reported primary evaluation cohort sizes, and selected headline metrics. It does not rerun machine-learning training.

## Recompute the analyses

The broad workflow is:

1. Acquire the cited source records listed in `DATA_SOURCES.md`.
2. Construct the candidate plant list, reconciled bee cohort, and evidence ledger with the corresponding build scripts.
3. Rebuild the weekly flowering and native-bee timing panels.
4. Run the component-model comparisons, then the interaction and mixture replays.
5. Recompute uncertainty intervals and compare the regenerated compact outputs with the committed artifacts.

The paper experiment registry maps each research question to its input and output paths. The raw input snapshots are excluded because they are large and retain the original providers' terms. Source acquisition and rebuilding can require credentials for services such as NEON. No credentials are included in this repository.

## Determinism and limits

Random seeds are declared in the released analysis scripts. Hardware and library differences can produce minor floating-point variation, especially for neural models. The archived compact artifacts are the reference values for the manuscript. Results should be interpreted within the study's stated limits. They evaluate retrospective seasonal overlap and ranking performance, not causal ecological or agricultural outcomes.
