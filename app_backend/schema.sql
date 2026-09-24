PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    ingestion_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at_utc TEXT NOT NULL,
    completed_at_utc TEXT,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    pipeline_version TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS data_releases (
    release_id TEXT PRIMARY KEY,
    methodology_version TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    published_at_utc TEXT,
    status TEXT NOT NULL CHECK (status IN ('working', 'validated', 'published', 'retired')),
    claim_boundary TEXT NOT NULL,
    ingestion_id INTEGER NOT NULL REFERENCES ingestion_runs(ingestion_id),
    is_current INTEGER NOT NULL DEFAULT 0 CHECK (is_current IN (0, 1))
);

CREATE UNIQUE INDEX IF NOT EXISTS one_current_release
ON data_releases(is_current) WHERE is_current = 1;

CREATE TABLE IF NOT EXISTS dataset_sources (
    source_id INTEGER PRIMARY KEY AUTOINCREMENT,
    release_id TEXT NOT NULL REFERENCES data_releases(release_id),
    role TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    UNIQUE(release_id, role, relative_path)
);

CREATE TABLE IF NOT EXISTS states (
    state_code TEXT PRIMARY KEY,
    state_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS grid_cells (
    release_id TEXT NOT NULL REFERENCES data_releases(release_id),
    state_code TEXT NOT NULL REFERENCES states(state_code),
    cell_id TEXT NOT NULL,
    resolution_degrees REAL,
    center_latitude REAL NOT NULL,
    center_longitude REAL NOT NULL,
    PRIMARY KEY(release_id, cell_id)
);

CREATE INDEX IF NOT EXISTS grid_cells_state_location
ON grid_cells(release_id, state_code, center_latitude, center_longitude);

CREATE TABLE IF NOT EXISTS plants (
    plant_id INTEGER PRIMARY KEY AUTOINCREMENT,
    scientific_name TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS state_plants (
    release_id TEXT NOT NULL REFERENCES data_releases(release_id),
    state_code TEXT NOT NULL REFERENCES states(state_code),
    plant_id INTEGER NOT NULL REFERENCES plants(plant_id),
    planting_class TEXT NOT NULL,
    coarse_habitat TEXT,
    state_native INTEGER NOT NULL CHECK (state_native IN (0, 1)),
    state_exotic INTEGER NOT NULL CHECK (state_exotic IN (0, 1)),
    state_rank TEXT,
    state_regulated INTEGER NOT NULL CHECK (state_regulated IN (0, 1)),
    eligible INTEGER NOT NULL CHECK (eligible IN (0, 1)),
    exclusion_reason TEXT,
    phenology_confidence REAL,
    hydrology_constraint_available INTEGER CHECK (hydrology_constraint_available IN (0, 1)),
    constraint_warnings TEXT,
    native_evidence_source TEXT,
    legal_evidence_source TEXT,
    PRIMARY KEY(release_id, state_code, plant_id)
);

CREATE TABLE IF NOT EXISTS bees (
    bee_id INTEGER PRIMARY KEY AUTOINCREMENT,
    scientific_name TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    family TEXT,
    bee_group TEXT,
    is_specialist INTEGER CHECK (is_specialist IN (0, 1)),
    is_bumble INTEGER CHECK (is_bumble IN (0, 1))
);

CREATE TABLE IF NOT EXISTS state_bees (
    release_id TEXT NOT NULL REFERENCES data_releases(release_id),
    state_code TEXT NOT NULL REFERENCES states(state_code),
    bee_id INTEGER NOT NULL REFERENCES bees(bee_id),
    is_native INTEGER NOT NULL CHECK (is_native IN (0, 1)),
    eligible INTEGER NOT NULL CHECK (eligible IN (0, 1)),
    occurrence_records INTEGER,
    sampling_events INTEGER,
    observed_years INTEGER,
    observed_weeks INTEGER,
    PRIMARY KEY(release_id, state_code, bee_id)
);

CREATE TABLE IF NOT EXISTS bee_activity (
    release_id TEXT NOT NULL REFERENCES data_releases(release_id),
    state_code TEXT NOT NULL REFERENCES states(state_code),
    bee_id INTEGER NOT NULL REFERENCES bees(bee_id),
    iso_week INTEGER NOT NULL CHECK (iso_week BETWEEN 1 AND 53),
    activity_mass REAL NOT NULL,
    relative_activity REAL,
    PRIMARY KEY(release_id, state_code, bee_id, iso_week)
);

CREATE TABLE IF NOT EXISTS phenology_predictions (
    release_id TEXT NOT NULL REFERENCES data_releases(release_id),
    state_code TEXT NOT NULL REFERENCES states(state_code),
    cell_id TEXT NOT NULL,
    plant_id INTEGER NOT NULL REFERENCES plants(plant_id),
    iso_year INTEGER NOT NULL,
    iso_week INTEGER NOT NULL CHECK (iso_week BETWEEN 1 AND 53),
    flowering_probability REAL NOT NULL CHECK (flowering_probability BETWEEN 0 AND 1),
    confidence REAL,
    provenance TEXT,
    PRIMARY KEY(release_id, cell_id, plant_id, iso_year, iso_week),
    FOREIGN KEY(release_id, cell_id) REFERENCES grid_cells(release_id, cell_id)
);

CREATE INDEX IF NOT EXISTS phenology_lookup
ON phenology_predictions(release_id, cell_id, plant_id, iso_week);

CREATE TABLE IF NOT EXISTS plant_suitability (
    release_id TEXT NOT NULL REFERENCES data_releases(release_id),
    state_code TEXT NOT NULL REFERENCES states(state_code),
    cell_id TEXT NOT NULL,
    plant_id INTEGER NOT NULL REFERENCES plants(plant_id),
    suitability_score REAL NOT NULL,
    local_multiplier REAL,
    local_records INTEGER,
    interpretation TEXT,
    PRIMARY KEY(release_id, cell_id, plant_id),
    FOREIGN KEY(release_id, cell_id) REFERENCES grid_cells(release_id, cell_id)
);

CREATE TABLE IF NOT EXISTS plant_bee_links (
    release_id TEXT NOT NULL REFERENCES data_releases(release_id),
    state_code TEXT NOT NULL REFERENCES states(state_code),
    plant_id INTEGER NOT NULL REFERENCES plants(plant_id),
    bee_id INTEGER NOT NULL REFERENCES bees(bee_id),
    evidence_type TEXT NOT NULL CHECK (evidence_type IN ('documented', 'predicted')),
    probability_center REAL NOT NULL,
    probability_low REAL NOT NULL,
    probability_high REAL NOT NULL,
    uncertainty_label TEXT,
    abstain_from_direct_claim INTEGER NOT NULL CHECK (abstain_from_direct_claim IN (0, 1)),
    model_disagreement REAL,
    admission_layer TEXT,
    use_restriction TEXT,
    PRIMARY KEY(release_id, state_code, plant_id, bee_id)
);

CREATE INDEX IF NOT EXISTS link_plant_lookup
ON plant_bee_links(release_id, state_code, plant_id, evidence_type);

CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    release_id TEXT NOT NULL REFERENCES data_releases(release_id),
    state_code TEXT NOT NULL REFERENCES states(state_code),
    cell_id TEXT NOT NULL,
    planting_class TEXT NOT NULL,
    policy TEXT NOT NULL,
    decision_status TEXT NOT NULL,
    change_gate_passed INTEGER NOT NULL CHECK (change_gate_passed IN (0, 1)),
    improvement_probability REAL NOT NULL,
    policy_gain_p10 REAL NOT NULL,
    availability_gain_p10 REAL NOT NULL,
    minimum_plant_stability REAL NOT NULL,
    score_mean REAL NOT NULL,
    score_p10 REAL NOT NULL,
    availability_mean REAL NOT NULL,
    availability_p10 REAL NOT NULL,
    claim_boundary TEXT NOT NULL,
    ai_gate_open INTEGER NOT NULL CHECK (ai_gate_open IN (0, 1)),
    ai_gate_reason TEXT,
    UNIQUE(release_id, cell_id, planting_class),
    FOREIGN KEY(release_id, cell_id) REFERENCES grid_cells(release_id, cell_id)
);

CREATE TABLE IF NOT EXISTS recommendation_plants (
    recommendation_id INTEGER NOT NULL REFERENCES recommendations(recommendation_id) ON DELETE CASCADE,
    plant_id INTEGER NOT NULL REFERENCES plants(plant_id),
    ordinal INTEGER NOT NULL,
    PRIMARY KEY(recommendation_id, plant_id),
    UNIQUE(recommendation_id, ordinal)
);

CREATE TABLE IF NOT EXISTS model_artifacts (
    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    release_id TEXT NOT NULL REFERENCES data_releases(release_id),
    component TEXT NOT NULL,
    version TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    operational INTEGER NOT NULL CHECK (operational IN (0, 1)),
    UNIQUE(release_id, component, relative_path)
);

CREATE TABLE IF NOT EXISTS validation_results (
    validation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    release_id TEXT NOT NULL REFERENCES data_releases(release_id),
    check_name TEXT NOT NULL,
    passed INTEGER NOT NULL CHECK (passed IN (0, 1)),
    observed_value TEXT,
    expected_value TEXT,
    details TEXT,
    UNIQUE(release_id, check_name)
);

CREATE TABLE IF NOT EXISTS farmer_feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    recommendation_id INTEGER REFERENCES recommendations(recommendation_id),
    submitted_at_utc TEXT NOT NULL,
    location_latitude REAL,
    location_longitude REAL,
    feedback_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    verification_status TEXT NOT NULL DEFAULT 'unreviewed'
        CHECK (verification_status IN ('unreviewed', 'accepted', 'rejected')),
    reviewer_notes TEXT
);

CREATE VIEW IF NOT EXISTS current_recommendations AS
SELECT r.*
FROM recommendations r
JOIN data_releases d ON d.release_id = r.release_id
WHERE d.is_current = 1 AND d.status = 'published';

INSERT OR REPLACE INTO schema_metadata(key, value) VALUES ('schema_version', '1');
