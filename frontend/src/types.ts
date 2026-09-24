export interface ReleaseInfo {
  release_id: string;
  methodology_version: string;
  published_at_utc: string;
}

export interface HealthResponse {
  status: "ready";
  service: string;
  release: ReleaseInfo;
}

export interface StateSummary {
  state_code: string;
  state_name: string;
  modeled_cells: number;
  published_cells: number;
  guidance_level: "local_release" | "localized_with_statewide_fallback" | "statewide_only" | "unavailable";
}

export interface PlantingClassSummary {
  planting_class: string;
  modeled_cells: number;
  mixture_size: number;
}

export interface GridCell {
  cell_id: string;
  center_latitude: number;
  center_longitude: number;
  resolution_degrees: number;
  has_published_recommendation: boolean;
  cell_validation_pass: boolean | null;
  evidence_grade: "strong_local" | "modeled_limited" | "modeled_no_recommendation";
  county_name: string | null;
  state_area_fraction: number;
}

export interface BoundaryCollection {
  type: "FeatureCollection";
  features: Array<{
    type: "Feature";
    properties: { kind: "state" | "county"; BASENAME: string; GEOID: string; [key: string]: unknown };
    geometry: { type: string; coordinates: unknown };
  }>;
}

export interface EvidenceCoverage {
  plant_occurrence_records: number | null;
  flowering_observations: number | null;
  flowering_observation_years: number | null;
  model_years: number | null;
  documented_state_network_links: number | null;
  bee_occurrence_records: number | null;
  bee_sampling_events: number | null;
  bee_observation_years: number | null;
  native_bee_species: number | null;
  cell_validation_pass: boolean | null;
  bee_coverage_source: string | null;
}

export interface BeeGroupDetail {
  bee_group: string;
  bees: string[];
}

export interface RecommendedPlant {
  scientific_name: string;
  display_name: string;
  planting_class: string;
  coarse_habitat: string | null;
  local_suitability_score: number | null;
  suitability_interpretation: string | null;
  phenology_confidence: number | null;
  modeled_peak_flowering_week: number | null;
  modeled_weeks_at_or_above_20_percent: number[];
  documented_eligible_native_bee_species_count: number;
  documented_native_bee_groups: BeeGroupDetail[];
  common_name: string | null;
  growth_form: string | null;
  season_group: string | null;
  moisture_guidance: string | null;
  light_guidance: string | null;
  acidic_preference: boolean | null;
  guidance_status: string | null;
  guidance_source: string | null;
  practical_note: string | null;
  image: {
    local_path: string | null;
    source_url: string | null;
    creator: string | null;
    license: string | null;
    title: string | null;
  } | null;
}

export interface RecommendationResponse {
  release: ReleaseInfo;
  request: {
    state_code: string;
    planting_class: string;
    requested_cell_id: string | null;
    requested_latitude: number | null;
    requested_longitude: number | null;
    resolved_cell_id: string;
    modeled_cell_center: [number, number];
    distance_to_modeled_cell_km: number;
  };
  spatial_decision: {
    policy_version: string;
    level: "local_grid" | "county" | "ecological_region" | "statewide";
    evidence_grade: "strong_local" | "supported_fallback" | "statewide_guidance";
    label: string;
    county_name: string | null;
    target_cell_id: string;
    source_cell_id: string;
    distance_km: number;
    county_limit_km: number;
    ecological_limit_km: number;
    environmental_similarity: number | null;
    environment_features_compared: number;
    reason: string;
  };
  local_evidence: EvidenceCoverage;
  source_evidence: EvidenceCoverage;
  recommendation: {
    plants: RecommendedPlant[];
    mixture_size: number;
    decision_status: string;
    robust_change_accepted: boolean;
    improvement_probability: number;
    lower_tail_availability_score: number;
    minimum_candidate_plant_stability: number;
    policy: string;
    ai_used_operationally: boolean;
    ai_gate_reason: string | null;
  };
  documented_native_bee_support: Array<{
    bee_group: string;
    species_count: number;
    species: string[];
  }>;
  interpretation: {
    confidence_signal: string;
    claim_boundary: string;
    important_note: string;
  };
}

export interface ApiErrorBody {
  error?: string;
  message?: string;
  detail?: unknown;
}

export interface SubmissionReceipt {
  identifier: string;
  status: "recorded_unverified";
  message: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatContext {
  state_code?: string;
  planting_class?: string;
  latitude?: number;
  longitude?: number;
}

export interface ChatResponse {
  message: string;
  model: string;
  tools_used: Array<{ name: string }>;
  recommendation: RecommendationResponse | null;
  claim_boundary: string;
}

export interface ChatStatusResponse {
  available: boolean;
  model: string;
  role: string;
}

export interface ResearchValidationMethod {
  method: string;
  label: string;
  bee_week_coverage: number;
  bee_coverage: number;
  week_coverage: number;
  difference_vs_documented_links: number | null;
  difference_ci95: [number, number] | null;
}

export interface ResearchValidationView {
  design: string;
  label: string;
  methods: ResearchValidationMethod[];
}

export interface LatestResearchValidation {
  research_release: string;
  status: string;
  operational_status: string;
  source_directory: string;
  validation_designs: Record<string, string>;
  link_layers: string[];
  mixture_size_displayed: number;
  views: ResearchValidationView[];
  not_claimed: string;
}
