# Pollinator Forage Feasibility Audit

This directory contains the reproducible first phase of the pollinator forage-gap project.

The current goal is to test real dataset access, inspect schemas, and measure overlap before model development.

## Run

```powershell
python scripts/download_feasibility.py
python scripts/summarize_feasibility.py
```

The download script uses only the Python standard library.

Raw responses are stored under `data/raw/<source>/`.

Each source receives a metadata record containing its request URL, access time, status, byte count, and SHA-256 checksum.

## Current scope

- Pennsylvania.
- August 2024 for the initial USA-NPN schema sample.
- Up to 1,200 Pennsylvania USGS Bee Database records from GBIF.
- A representative Pennsylvania Daymet point for 2024 (currently blocked by HTTP 503).
- Bee-Gap attachments available through ScienceBase.
- Seed plant genera for small GloBI interaction samples.
- Metadata checks for NEON and other supporting sources.

This is an audit sample, not the final research dataset.

## First-stage outputs

- `reports/INITIAL_FEASIBILITY_AUDIT.txt`: concise findings and limitations.
- `data/interim/dataset_inventory.csv`: access, fields, coverage, and missingness.
- `data/interim/initial_plant_overlap.csv`: phenology, interaction, host, and USDA matches.
- `data/interim/initial_bee_overlap.csv`: occurrence, Bee-Gap traits, and provisional guilds.
- `data/interim/usda_candidate_plants.csv`: resolved candidate profiles and native-status flags.

NEON data files require a free API token. The downloaded NEON product metadata is enough to verify the product, but not to perform validation yet.

## Target-selection outputs

- `reports/TARGET_LIST_FOR_STUDENT.txt`: concise plant and bee research panel.
- `reports/TARGET_LIST_JUSTIFICATION.txt`: evidence rules, results, and limitations.
- `data/interim/target_plants_ranked.csv`: 24 primary plants plus six alternates.
- `data/interim/target_bees_ranked.csv`: focal bee species, traits, counts, and modeling roles.
- `scripts/download_target_evidence.py`: retrieves Pennsylvania-native status, 2015-2025 NPN phenometrics, full bee frequency facets, and candidate GloBI evidence.
- `scripts/build_target_lists.py`: applies the transparent selection rules and regenerates the target files.

## Weekly pilot dataset

Run in this order:

```powershell
python scripts/download_weekly_sources.py
python scripts/build_weekly_coverage.py
python scripts/download_nasa_weather.py
python scripts/build_weekly_panel.py
python scripts/validate_weekly_panel.py
```

Main outputs:

- `data/interim/weekly/weekly_modeling_panel_statewide.csv`: dense 2015-2025 panel for seven plants and ten bees. Blank observation fields mean unsampled/unknown, never absence.
- `data/interim/weekly/plant_weekly_statewide.csv`: observed weekly flowering summaries.
- `data/interim/weekly/bee_weekly_statewide.csv`: observed bee occurrence and sampling-event summaries.
- `data/interim/weekly/spatial_coverage_comparison.csv`: statewide versus 1-degree and 0.5-degree coverage.
- `reports/WEEKLY_COVERAGE_AUDIT.txt`: evidence supporting the statewide pilot resolution.
- `reports/WEEKLY_DATASET_BUILD_REPORT.txt`: panel construction summary.
- `reports/WEEKLY_PANEL_VALIDATION.txt`: integrity-test result.

Daymet's public single-pixel service returned HTTP 503 during this build. The completed panel therefore uses NASA POWER daily weather averaged across the occupied one-degree Pennsylvania buckets. NPN-attached Daymet fields remain in the plant aggregates as an independent secondary weather source where available.

## Statewide baseline model

Run after the weekly pilot dataset is complete:

```powershell
python scripts/build_bee_event_effort.py
python scripts/fit_statewide_baselines.py
python scripts/build_statewide_gap_index.py
python scripts/summarize_statewide_baseline.py
python scripts/validate_statewide_baseline.py
```

Main outputs are under `results/statewide_baseline/`. The plant test uses future years 2023-2025. The bee test uses 2023-2024 and reconstructs valid focal-species non-detections only inside documented USGS bee-positive sampling events. Bee activity outside weeks 11-41 is deliberately left unsupported.

The file `target_panel_gap_index.csv` describes only the seven modeled plants. It must not be interpreted as total forage, nutrition, abundance, health, or a verified local shortage.

## Citation-traceable plant-bee evidence audit

Run after the statewide baseline:

```powershell
python scripts/download_interaction_provenance.py
python scripts/build_interaction_evidence_audit.py
python scripts/rerun_gap_evidence_sensitivity.py
python scripts/summarize_interaction_audit.py
python scripts/validate_interaction_audit.py
```

The audit queries all 70 plant-bee pairs in both directions, preserves the detailed GloBI records, deduplicates mirrored imports, adds exact Bee-Gap host records, and assigns evidence grades A, B, C, or U. The primary gap model should use A/B links. Grade C is for sensitivity analysis, and U means unknown rather than impossible.

## Expanded ten-plant pilot

This stage adds New England aster, Virginia mountain mint, and boneset. It also changes the phenology outcome to open flowers only and adds a GBIF occurrence-support sensitivity layer.

```powershell
python scripts/download_expanded_plant_sources.py
python scripts/download_interaction_provenance.py --expanded
python scripts/build_interaction_evidence_audit.py --expanded
python scripts/build_expanded_plant_dataset.py
python scripts/fit_expanded_pooled_flowering.py
python scripts/compare_expanded_plant_mixtures.py
python scripts/summarize_expanded_plant_pilot.py
python scripts/validate_expanded_plant_pilot.py
```

The occurrence-support index measures documentation breadth across counties, grid cells, and years. It is not plant abundance or a formal occupancy probability. The mixture experiment exhaustively compares all 120 seven-plant subsets of the ten candidates under unweighted, occurrence-supported, and Pennsylvania-direct interaction scenarios.

## Evidence-aware Pennsylvania localization experiment

Run after the expanded ten-plant pilot:

```powershell
python -m pip install -r requirements-localization.txt
python scripts/download_pa_grid_environment.py
python scripts/build_pa_ecological_grid.py
python scripts/run_localized_recommendation_experiment.py
python scripts/validate_localized_recommendations.py
```

The experiment builds 59 half-degree Pennsylvania cells with 2015-2025 NASA POWER weather, sampled USGS 3DEP elevation, sampled USDA SSURGO soil, 2025 USDA Cropland Data Layer classes, NPN open-flower observations, and effort-shrunk GBIF plant documentation. It tests localization using future observations in geographically withheld cells.

Main outputs:

- `data/interim/localization/pa_ecological_grid_features.csv`: environmental cell features.
- `results/localized_recommendations/localized_model_validation_metrics.csv`: strict statewide-versus-localized prediction test.
- `results/localized_recommendations/grid_localized_recommendations.csv`: one model-based seven-plant result per cell.
- `reports/LOCALIZED_RECOMMENDATION_RESULTS.txt`: concise scientific findings.
- `reports/LOCALIZED_RECOMMENDATION_VALIDATION.txt`: reproducibility and integrity checks.

Current result: unrestricted localization predicts unseen regions worse than the statewide flowering model. An evidence gate therefore falls back fully to statewide phenology. Small mixture changes remain possible through local plant documentation, but they are not field-validated and should not yet be delivered as farm recommendations. Soil and land cover describe environmental transfer; species-specific planting tolerances remain future work.

## Authenticated NEON and USDA NASS sources

NEON is used as an external phenology validation source at the three closest suitable
Mid-Atlantic sites (BLAN, SCBI, and SERC). It is not treated as Pennsylvania-local
ground truth. NASS Quick Stats supplies Pennsylvania state and county agricultural
context for eight pollination-relevant crop groups; it does not measure floral forage.

Copy `.secrets.example.json` to `.secrets.json`, insert the two locally issued values,
and never commit that file. Environment variables named `NEON_API_TOKEN` and
`NASS_API_KEY` take precedence over the local secrets file.

```powershell
Copy-Item .secrets.example.json .secrets.json
python scripts\download_authenticated_sources.py --check
python scripts\download_authenticated_sources.py
python scripts\summarize_authenticated_sources.py
```

The downloader redacts keys from its provenance manifest. Raw files are stored under
`data/raw/neon_phenology/` and `data/raw/nass_quickstats/`.

The derived NEON transportability test and NASS county/grid context are built with:

```powershell
python scripts\download_derived_component_sources.py
python scripts\build_derived_components.py
python scripts\validate_derived_components.py
```

NEON is explicitly treated as a geographic stress test. The Pennsylvania model is
not expected to reproduce identical regional timing; the test asks whether its
weather adjustment transports better than an unadjusted Pennsylvania calendar.
NASS acreage remains a documented lower bound whenever values are suppressed.
# Hierarchical localization experiment

The hierarchical experiment tests whether Pennsylvania-wide flowering patterns can be
localized safely through partially pooled ecological-region corrections. Model settings
are selected with nested 2022 validation inside the training geography. The final test
uses 2023-2025 observations from geographically excluded cells.

```powershell
python scripts\run_hierarchical_localization_experiment.py
python scripts\validate_hierarchical_localization.py
```

Primary outputs are written to `results/hierarchical_localization/`. The concise result
and validation reports are `reports/HIERARCHICAL_LOCALIZATION_RESULTS.txt` and
`reports/HIERARCHICAL_LOCALIZATION_VALIDATION.txt`.

## Hierarchical mixture experiment

This experiment sends the geographically held-out statewide and hierarchical flowering
forecasts into the seven-of-ten plant-mixture engine. It compares the mixture preferred
under each forecast with a consensus mixture that minimizes the worse objective across
both phenology models.

```powershell
python scripts\run_hierarchical_mixture_experiment.py
python scripts\validate_hierarchical_mixture_experiment.py
```

Outputs are written to `results/hierarchical_mixtures/`, with concise reports in
`reports/HIERARCHICAL_MIXTURE_RESULTS.txt` and
`reports/HIERARCHICAL_MIXTURE_VALIDATION.txt`.

## Crop-pollinator bridge

This component links the county crop context to sourced crop timing and documented
pollinator groups. It deliberately preserves unknown timing instead of treating a
missing record as no bloom. Apple and peach use USDA NASS weekly full-bloom progress;
USA-NPN crop observations are retained as sparse proxies; Pennsylvania cucurbit field
evidence supplies a visitation sampling window. The resulting county records localize
crop acreage, not yet county-specific bloom or pollinator demand.

```powershell
python scripts\download_crop_pollination_bridge.py
python scripts\build_crop_pollination_bridge.py
python scripts\validate_crop_pollination_bridge.py
```

Main outputs are in `data/interim/crop_pollination_bridge/`. The short scientific
summary is `reports/CROP_POLLINATION_BRIDGE_RESULTS.txt`. Dryad file downloads may
return HTTP 401 even though the public API exposes their metadata; the downloader
records this source limitation explicitly.

The bridge can be tested inside the plant-mixture engine with a constrained
multi-objective experiment:

```powershell
python scripts\run_crop_aware_mixture_experiment.py
python scripts\validate_crop_aware_mixture_experiment.py
```

The crop-aware engine minimizes modeled crop-week forage gaps subject to a maximum
allowed loss in the original season-long pollinator objective. It tests 0%, 2%, and
5% limits and separates direct USDA crop-progress evidence from a sensitivity scenario
that adds down-weighted sparse NPN and Pennsylvania visitation-window evidence.

## Hybrid pollination simulator

This reduced-order stochastic simulator connects weekly flowering, floral reward,
plant-bee evidence, bee activity and resource memory, crop bloom, acreage, and
crop-versus-habitat foraging allocation. It compares no planting, a Pennsylvania
consensus mixture, the general gap mixture, and the localized crop-aware mixture.

Run in this order:

```powershell
python scripts\download_hybrid_simulator_sources.py
python scripts\build_hybrid_simulator_inputs.py
python scripts\run_hybrid_pollination_simulator.py
python scripts\validate_hybrid_pollination_simulator.py
```

The simulation uses local, checksummed snapshots rather than live APIs. It runs 200
uncertainty draws across 19 cells and the 2023-2025 held-out weather years. Floral
reward transfers, establishment failure, and structural bee-state assumptions are
explicit. BEE-STEWARD V2 is retained as an open mechanistic reference; this Python
prototype does not claim to reproduce BEE-STEWARD.

Main outputs are in `results/hybrid_simulator/`. Read
`reports/HYBRID_SIMULATOR_RESULTS.txt` first. All crop and bee outputs are relative
indices for scenario comparison. They are not absolute bee counts, causal yield
effects, or field-validated prescriptions.

## Phase-aware recommendation engine

The revised optimizer tests all 120 seven-of-ten mixtures. It uses the hybrid simulator
to distinguish pre-bloom bee support, during-bloom flower competition, and post-bloom
support. It produces separate apple, blueberry, peach, pumpkin, and whole-farm choices.

```powershell
python scripts\optimize_phase_aware_mixtures.py
python scripts\test_phase_aware_recommendations.py
python scripts\validate_phase_aware_optimizer.py
```

Example prototype lookup:

```powershell
python scripts\recommend_farm_mixture.py --latitude 40.04 --longitude -76.30 --crop apples
```

Read `reports/PHASE_AWARE_OPTIMIZER_RESULTS.txt` for the short result. The lookup uses
the nearest one of 19 modeled cells with complete crop evidence. Its output remains a
relative, coarse-grid research recommendation requiring field validation.

## AI evidence expansion and graph preparation

The phase-aware V1 model is locked before AI-assisted evidence expansion. The evidence
agent discovers literature through Europe PMC, OpenAlex, and Crossref, retrieves
available full text, identifies candidate passages, and ranks them with a local sentence
embedding model. It never accepts a measurement automatically.

```powershell
python scripts\freeze_research_baseline.py
python scripts\run_floral_evidence_agent.py
python scripts\triage_floral_evidence_with_embeddings.py
python scripts\build_temporal_graph_dataset.py
python scripts\validate_ai_evidence_foundation.py
```

Optional structured LLM extraction uses the OpenAI Responses API and requires the
`OPENAI_API_KEY` environment variable:

```powershell
python scripts\run_floral_evidence_agent.py --use-llm
```

LLM results are written only as review candidates. A human must confirm the exact quote,
species, metric, value, unit, and biological basis before a model prior can change.
## Unified PA–MD–MI engine (V30)

The V30 runner gives Pennsylvania, Maryland, and Michigan one state configuration contract and one native-bee scorecard implementation. It consumes the already-built state artifacts, preserves the frozen state decision gates, and verifies that released mixtures are reproduced. Michigan also carries a separate held-out prairie-visitation validation layer.

Run both configured states:

```powershell
python scripts\run_multistate_engine_v30.py
python scripts\validate_multistate_engine_v30.py
```

Run one state:

```powershell
python scripts\run_multistate_engine_v30.py --state PA
```

The master configuration is `config/multistate_engine_v30.json`. State adapters are under `config/states/`. Outputs are written to `results/multistate_engine_v30/`.

## Cross-state plant-bee evidence foundation (V31 candidate)

V31 builds one provenance-preserving plant-bee registry for state reuse. It treats an
original study or dataset as the source unit, quarantines ambiguous taxonomic aliases,
keeps undocumented pairs as unknown, and never promotes AI/GNN predictions to confirmed
edges. Maryland and Michigan outcome records remain excluded from evidence construction.

```powershell
python scripts\download_mangal_network_v31.py
python scripts\build_network_foundation_v31.py
python scripts\reconcile_network_taxa_v31.py
python scripts\audit_cross_state_network_gaps_v31.py
python scripts\audit_network_source_depletion_v31.py
python scripts\validate_network_foundation_v31.py
```

Read `reports/NETWORK_FOUNDATION_V31_RESULTS.txt` for the concise findings. V31 is a
candidate evidence layer; V30 remains the released decision engine until V31 is tested
on a fresh, untouched validation dataset.

## Evidence-grounded link generalization (V32 experiment)

V32 compares species-popularity, interpretable biological models, GraphSAGE, and a
biology-plus-GraphSAGE hybrid under leave-one-study-out evaluation. Predictions remain
separate from the V31 confirmed-evidence registry.

```powershell
python scripts\run_evidence_grounded_generalization_v32.py
python scripts\validate_evidence_grounded_generalization_v32.py
```

Read `reports/EVIDENCE_GROUNDED_GENERALIZATION_V32_RESULTS.txt`. Maryland and Michigan
are development diagnostics, not fresh independent validation, and trial-panel plant
rankings are not released farm recommendations.

## Published ecological-AI benchmarks (V33)

V33 tests the official pretrained eco_ILP model in an isolated pinned container and an
independent NECTAR-style taxonomy/graph-transfer implementation. Warm-start network
completion and cold-start state transfer are reported separately.

```powershell
python scripts\prepare_ecoilp_benchmark_v33.py
docker build -t pollinator-ecoilp-v33 -f external_benchmarks\ecoilp\Dockerfile external_benchmarks\ecoilp
powershell -ExecutionPolicy Bypass -File scripts\run_ecoilp_container_v33.ps1
python scripts\run_published_method_benchmarks_v33.py
python scripts\build_state_predicted_network_layer_v33.py
python scripts\validate_published_method_benchmarks_v33.py
```

Read `reports/PUBLISHED_ECOLOGICAL_AI_BENCHMARKS_V33_RESULTS.txt`. Predicted links are
stored in a separate research-candidate layer and never promoted to confirmed evidence.

## Model-ensemble comparison (V34)

V34 compares the V32 GraphSAGE-plus-biology model, the NECTAR-style model, their
mean-rank ensemble, and a conservative agreement ensemble on identical Maryland and
Michigan candidate universes and prediction budgets.

```powershell
python scripts\run_ensemble_comparison_v34.py
python scripts\validate_ensemble_comparison_v34.py
```

Read `reports/ENSEMBLE_COMPARISON_V34_RESULTS.txt`. This is a development diagnostic,
not an independent final validation, and predicted edges remain separate from evidence.

## Diversity-aware model portfolio (V35)

V35 tests whether a protected NECTAR exploration channel adds useful discoveries and
taxonomic diversity to the GraphSAGE ranking under an identical total link budget.

```powershell
python scripts\run_diversity_aware_portfolio_v35.py
python scripts\validate_diversity_aware_portfolio_v35.py
```

Read `reports/DIVERSITY_AWARE_PORTFOLIO_V35_RESULTS.txt`. The fixed portfolio is a
development hypothesis for fresh-state validation; predicted links are not evidence.

## Uncertainty-aware automatic recommendation layer (V36)

V36 calibrates the research link scores using out-of-study predictions, propagates
link, phenology, timing, and establishment uncertainty through mixture simulations,
and applies a conservative automatic-change gate. It returns one mixture plus core
plants, optional plants, substitutes, confidence, and a retain/change/review action.

```powershell
python scripts\run_uncertainty_calibration_v36.py
python scripts\run_uncertainty_aware_mixtures_v36.py
python scripts\validate_uncertainty_aware_engine_v36.py
```

Read `reports/UNCERTAINTY_AWARE_ENGINE_V36_RESULTS.txt`. V36 is not a production or
field-validated recommendation release.

## State plant-constraint and release audit (V37)

V37 audits the active panel against state-native distribution and current official
regulated-plant sources. It hard-filters ineligible plants before optimization,
requires visible review when site-suitability evidence is missing, and writes the
constraint provenance for every selected plant.

```powershell
python scripts\audit_plant_constraints_v37.py
python scripts\run_constrained_uncertainty_engine_v37.py
python scripts\validate_constrained_engine_v37.py
```

Read `reports/CONSTRAINED_ENGINE_V37_RESULTS.txt`. The audit fixed the obsolete PA
suitability-table reference and removed two plants from Michigan's native-only search
space. Soil and establishment requirements are not yet complete enough for operational
farm recommendations.

## Existing-data system evaluation (V38)

V38 reruns the Maryland common-garden and Michigan prairie evaluations and summarizes
them alongside the 18-study interaction benchmark and current V37 recommendation
coverage.

```powershell
python scripts\evaluate_maryland_common_garden_shadow_v26.py
python scripts\evaluate_prairie_independent_replication_v28.py
python scripts\evaluate_current_system_existing_data_v38.py
```

Read `reports/CURRENT_SYSTEM_EXISTING_DATA_V38_RESULTS.txt`. These datasets test the
interaction and visitation-ranking components, but they do not contain complete V37
recommended mixtures and therefore cannot validate exact mixture benefit.

## Availability-first network gate (V39 development test)

V39 tests a safe fallback policy on the Michigan prairie observations. Flower
availability remains primary, undocumented links cannot reduce a score, and a
leave-one-site-out evidence gate enables a network adjustment only after a 0.01 AUC
gain on the other sites.

```powershell
python scripts\test_availability_first_network_gate_v39.py
```

Read `reports/AVAILABILITY_FIRST_NETWORK_GATE_V39_RESULTS.txt`. The gate remained
closed in every held-site decision and preserved the stronger availability baseline.
This is development evidence because the Michigan outcomes had already been inspected.

## Availability-first three-way optimizer ablation (V40)

V40 compares availability-only, documented-network, and AI-augmented mixture policies
using identical constrained candidates and shared uncertainty scenarios. Network-guided
policies cannot select a candidate more than 2% below the best lower-tail availability.
AI predictions remain non-operational unless a state passes an independent added-value
gate.

```powershell
python scripts\run_availability_first_ablation_v40.py
python scripts\backtest_v40_pa_future_detections.py
python scripts\validate_availability_first_ablation_v40.py
```

Read `reports/AVAILABILITY_FIRST_ABLATION_V40_RESULTS.txt`. The operational research
output uses documented-network fallback in PA, MD, and MI; AI candidates are retained
for analysis but do not control recommendations.

# Application database (V40)

The research outputs are packaged into an immutable, application-facing SQLite snapshot. The generated database is stored at `data/app/pollinator_forage.sqlite3` and is intentionally excluded from version control because it can be rebuilt from hashed source artifacts.

Build the initial PA/MD/MI snapshot:

```powershell
python scripts/build_app_database_v40.py
```

Validate the database and its exact agreement with V40:

```powershell
python scripts/validate_app_database_v40.py
```

Query a recommendation directly:

```powershell
python scripts/query_app_recommendation.py --state PA --planting-class "herbaceous strip" --latitude 40.0 --longitude -75.3
```

Run the production FastAPI service locally:

```powershell
uvicorn app_backend.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the generated interactive API documentation.

The versioned application endpoints are:

- `GET /health`
- `GET /v1/states`
- `GET /v1/states/{state_code}/planting-classes`
- `GET /v1/states/{state_code}/cells`
- `GET /v1/states/{state_code}/boundaries`
- `GET /v1/recommendations?state=PA&planting_class=herbaceous%20strip&latitude=40.0&longitude=-75.3`

Application code should use `app_backend.RecommendationService` or these endpoints. It should not query research CSV files or database tables directly.

The Render Blueprint is `render.yaml`. See `DEPLOY_RENDER.md` for the deployment procedure. The deployment build verifies and expands the immutable database artifact from `deploy/`; no persistent disk is required for the read-only API.

## Responsive web interface

The TypeScript and React interface is under `frontend/`. It provides coordinate or modeled-cell selection, evidence-qualified plant mixtures, flowering timelines, native-bee scorecards, uncertainty signals, provenance, and claim boundaries.

The current interactive layer adds:

- An OpenStreetMap location picker showing modeled grid coverage.
- Official Census state/county boundaries and a green/yellow/gray evidence legend.
- Downloadable recommendation PDFs generated in the browser.
- Common names and coarse site-screening guidance for all 28 plants.
- Locally cached Creative Commons plant photographs with creator, license, and source attribution.
- Anonymous recommendation feedback and follow-up outcome recording.

Feedback and outcomes are stored separately as unverified reports. They never become scientific evidence or model-training data without a later review process.

## Spatial decision hierarchy

The application no longer treats the geographically nearest modeled cell as automatically local. It resolves a coordinate in this order:

1. Exact 0.5-degree grid with a published recommendation.
2. A published grid in the same county, limited to 80 km.
3. A published grid in the same modeled ecological cluster, limited to 160 km and ranked by standardized similarity across measured climate, soil, elevation, and land-cover features.
4. The modal published mixture for the state and planting class, explicitly labeled statewide guidance.

The thresholds and rationale are versioned in `config/spatial_decision_policy.json`. Cell-level plant and bee counts remain nullable when the corresponding audit data are unavailable. Missing coverage is never interpreted as zero. Boundary and cell context are generated by `scripts/build_spatial_context.py` and stored under `data/reference/` with source and checksum metadata.

```powershell
cd frontend
npm install
npm run dev
```

Set `VITE_API_BASE_URL` to the FastAPI service URL in production. The Render Blueprint deploys the interface as a static site and the API as a separate web service.

## Rolling-data rule

The schema supports immutable releases, source hashes, ingestion runs, model artifacts, validation results, and separately quarantined farmer feedback. New records should enter a new working release. The current published release remains active until the new release passes validation and is explicitly published. Models and recommendations are therefore updated in controlled batches, not during a farmer request.

V40 remains a potential-forage recommendation system. Its outputs do not establish bee health, abundance, survival, crop yield, plant establishment, or causal field benefit.
