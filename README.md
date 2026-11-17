# ForagePath

ForagePath is an evidence-governed decision-support system for identifying native-plant mixtures that can provide seasonal forage support for native bees in the United States. It integrates weekly flowering forecasts, relative seasonal native-bee occurrence timing, traceable plant-bee interaction evidence, and spatial context.

The accompanying manuscript is **ForagePath: Evidence-governed Forecasting of Seasonal Pollinator Forage Support and Habitat Mixtures from Heterogeneous Ecological Records**.

## What this repository contains

The repository has two deliberately separated layers.

- **Released application.** The FastAPI service and React interface return precomputed, versioned guidance. The operational recommendation layer uses only direct, source-traceable Grade A and Grade B plant-bee evidence. It does not turn model-predicted links into documented ecological interactions.
- **Paper reproducibility package.** The `reproducibility/` directory documents the data sources, frozen cohort definitions, analysis code, compact result artifacts, and exact checks used to support the manuscript. The strict interaction ledger and reconciled candidate lists are released under `data/reproducibility/`.

Raw third-party records are not redistributed here. They remain subject to the source providers' terms and are obtained using the documented source-specific workflows.

## Quick start

### Run the released API locally

```powershell
python -m pip install -r requirements-app.txt
python scripts/prepare_render_database.py
uvicorn app_backend.api:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000/docs`.

### Run the web interface locally

```powershell
cd frontend
npm ci
npm run dev
```

Set `VITE_API_BASE_URL` to the API URL when the interface and API are served separately.

### Verify paper artifacts

```powershell
python -m pip install -r requirements-research.txt
python scripts/verify_paper_release.py
```

This verification reads the frozen compact artifacts that support the manuscript tables and checks their documented cohort sizes, key point estimates, and provenance counts. Full model retraining requires source-record acquisition and is described in [reproducibility/REPRODUCE.md](reproducibility/REPRODUCE.md).

## Evidence and claim boundary

ForagePath reports potential seasonal support based on the available evidence. It does not demonstrate floral abundance, bee abundance, bee health, survival, crop pollination, yield, plant establishment, or causal field benefit. A missing plant-bee record remains unknown, not biologically impossible.

## Repository guide

- [Reproducibility guide](reproducibility/REPRODUCE.md)
- [Data and provenance guide](reproducibility/DATA_SOURCES.md)
- [Paper experiment registry](results/paper_experiment_registry_v1/paper_experiment_registry.md)
- [Render deployment guide](DEPLOY_RENDER.md)
- [Citation metadata](CITATION.cff)

## Security

Do not commit API keys, downloaded credentials, browser profiles, or raw records with access restrictions. Local secrets belong in `.secrets.json` or `.env`, both of which are ignored by Git. See [SECURITY.md](SECURITY.md).
