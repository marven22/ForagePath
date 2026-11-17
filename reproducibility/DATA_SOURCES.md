# Data sources and provenance

ForagePath combines public ecological sources. The repository publishes derived, compact artifacts that are necessary to inspect the paper's results and decision evidence. It does not redistribute bulk source records when their size, license, or access conditions make that inappropriate.

| Source | Role in ForagePath | Access and provenance |
|---|---|---|
| USA National Phenology Network (USA-NPN) Status and Intensity API | Weekly flowering observations and candidate-plant phenology | `https://services.usanpn.org/npn_portal/observations/getObservations.json`. The frozen plant snapshot covers 2009-2025 and is described in `data/reproducibility/us_plant_phenology_snapshot_v1.json`. |
| Global Biodiversity Information Facility (GBIF) | Native-bee occurrence records used for relative seasonal timing | Records were retrieved from GBIF under its data-use conditions and cleaned into the documented 2009-2025 U.S. bee snapshot. The released cohort file contains no raw occurrence coordinates. |
| Bee-Gap | Native-bee target and trait reconciliation | Used to define the 436 target native bee species and trait-family grouping. |
| Global Biotic Interactions (GloBI), source literature, and traceable interaction archives | Plant-bee evidence ledger | Direct evidence was reconciled into the released provenance ledger. Grade A and B links are the only links eligible for operational recommendations. |
| USDA PLANTS | Native status and plant attributes | Used for candidate-plant reconciliation and explanatory attributes. |
| Daymet | Historical weather covariates for flowering-model evaluation | Historical site-week covariates were used in retrospective development and test rows. The released app does not request live weather at recommendation time. |
| USDA SSURGO, Cropland Data Layer, U.S. Census boundaries, EPA Level III ecoregions, and Canadian Ecological Framework layers | Environmental and geographic context | Used only for described spatial and screening context. Their source identifiers and derived values are retained in the appropriate release metadata. |
| NEON | Cross-source flowering evaluation only | Used as an external phenology evaluation source. It is not part of USA-NPN model fitting. Source access may require a user-issued token. |

## Released derived artifacts

- `data/reproducibility/candidate_native_flowering_plants.csv` contains the locked 1,138-plant candidate list.
- `data/reproducibility/national_bee_activity_cohorts.csv` contains the 436-bee evidence cohort assignment.
- `data/reproducibility/provenance_evidence_ledger.csv.gz` contains evidence provenance and Grade A, B, and C status.
- `data/reproducibility/strict_admitted_plant_bee_edges.csv` contains the 5,236 operational Grade A/B links.

Use the original providers' terms, attribution requirements, and access procedures when obtaining source data. The inclusion of a source in this workflow does not transfer its license.
