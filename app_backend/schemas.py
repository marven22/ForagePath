from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReleaseInfo(StrictModel):
    release_id: str
    methodology_version: str
    published_at_utc: str


class HealthResponse(StrictModel):
    status: Literal["ready"]
    service: str = "pollinator-forage-api"
    release: ReleaseInfo


class StateSummary(StrictModel):
    state_code: str
    state_name: str
    modeled_cells: int
    published_cells: int
    guidance_level: Literal["local_release", "localized_with_statewide_fallback", "statewide_only", "unavailable"] = "local_release"


class StateListResponse(StrictModel):
    states: list[StateSummary]


class PlantingClassSummary(StrictModel):
    planting_class: str
    modeled_cells: int
    mixture_size: int


class PlantingClassListResponse(StrictModel):
    state_code: str
    planting_classes: list[PlantingClassSummary]


class GridCell(StrictModel):
    cell_id: str
    center_latitude: float
    center_longitude: float
    resolution_degrees: float
    has_published_recommendation: bool
    cell_validation_pass: bool | None = None
    evidence_grade: Literal["strong_local", "modeled_limited", "modeled_no_recommendation"]
    county_name: str | None = None
    state_area_fraction: float = Field(ge=0, le=1)


class GridCellListResponse(StrictModel):
    state_code: str
    cells: list[GridCell]


class BeeGroupDetail(StrictModel):
    bee_group: str
    bees: list[str]


class PlantImage(StrictModel):
    local_path: str | None = None
    source_url: str | None = None
    creator: str | None = None
    license: str | None = None
    title: str | None = None


class RecommendedPlant(StrictModel):
    scientific_name: str
    display_name: str
    planting_class: str
    coarse_habitat: str | None = None
    local_suitability_score: float | None = None
    suitability_interpretation: str | None = None
    phenology_confidence: float | None = None
    modeled_peak_flowering_week: int | None = None
    modeled_weeks_at_or_above_20_percent: list[int]
    documented_eligible_native_bee_species_count: int
    documented_native_bee_groups: list[BeeGroupDetail]
    common_name: str | None = None
    growth_form: str | None = None
    season_group: str | None = None
    moisture_guidance: str | None = None
    light_guidance: str | None = None
    acidic_preference: bool | None = None
    guidance_status: str | None = None
    guidance_source: str | None = None
    practical_note: str | None = None
    image: PlantImage | None = None


class RequestResolution(StrictModel):
    state_code: str
    planting_class: str
    requested_cell_id: str | None = None
    requested_latitude: float | None = None
    requested_longitude: float | None = None
    resolved_cell_id: str
    modeled_cell_center: tuple[float, float]
    distance_to_modeled_cell_km: float


class SpatialDecision(StrictModel):
    policy_version: str
    level: Literal["local_grid", "county", "ecological_region", "statewide"]
    evidence_grade: Literal["strong_local", "local_replay_supported", "supported_fallback", "statewide_guidance"]
    label: str
    county_name: str | None = None
    target_cell_id: str
    source_cell_id: str
    distance_km: float = Field(ge=0)
    county_limit_km: float = Field(gt=0)
    ecological_limit_km: float = Field(gt=0)
    environmental_similarity: float | None = Field(default=None, ge=0, le=1)
    environment_features_compared: int = Field(ge=0)
    reason: str


class EvidenceCoverage(StrictModel):
    plant_occurrence_records: int | None = Field(default=None, ge=0)
    flowering_observations: int | None = Field(default=None, ge=0)
    flowering_observation_years: int | None = Field(default=None, ge=0)
    model_years: int | None = Field(default=None, ge=0)
    documented_state_network_links: int | None = Field(default=None, ge=0)
    bee_occurrence_records: int | None = Field(default=None, ge=0)
    bee_sampling_events: int | None = Field(default=None, ge=0)
    bee_observation_years: int | None = Field(default=None, ge=0)
    native_bee_species: int | None = Field(default=None, ge=0)
    cell_validation_pass: bool | None = None
    bee_coverage_source: str | None = None


class RecommendationResult(StrictModel):
    plants: list[RecommendedPlant]
    mixture_size: int
    decision_status: str
    robust_change_accepted: bool
    improvement_probability: float = Field(ge=0, le=1)
    lower_tail_availability_score: float
    minimum_candidate_plant_stability: float = Field(ge=0, le=1)
    policy: str
    ai_used_operationally: bool
    ai_gate_reason: str | None = None


class BeeSupportSummary(StrictModel):
    bee_group: str
    species_count: int
    species: list[str]


class Interpretation(StrictModel):
    confidence_signal: str
    claim_boundary: str
    important_note: str


class RecommendationResponse(StrictModel):
    release: ReleaseInfo
    request: RequestResolution
    spatial_decision: SpatialDecision
    local_evidence: EvidenceCoverage
    source_evidence: EvidenceCoverage
    recommendation: RecommendationResult
    documented_native_bee_support: list[BeeSupportSummary]
    interpretation: Interpretation


class ErrorBody(StrictModel):
    error: str
    message: str
    request_id: str


class ServiceIndex(StrictModel):
    service: str
    version: str
    documentation: str
    health: str


class FeedbackSubmission(StrictModel):
    release_id: str = Field(min_length=1, max_length=80)
    state_code: str = Field(min_length=2, max_length=2)
    cell_id: str = Field(min_length=1, max_length=80)
    planting_class: str = Field(min_length=1, max_length=80)
    selected_plants: list[str] = Field(min_length=1, max_length=20)
    helpful_rating: int = Field(ge=1, le=5)
    intended_action: Literal["considering", "planning", "planted", "not_using"]
    comments: str | None = Field(default=None, max_length=1500)


class OutcomeSubmission(StrictModel):
    feedback_id: str = Field(min_length=36, max_length=36)
    observation_date: date
    plants_attempted: list[str] = Field(min_length=1, max_length=20)
    establishment_fraction: float | None = Field(default=None, ge=0, le=1)
    flowering_observed: bool | None = None
    pollinators_observed: bool | None = None
    observed_groups: list[str] = Field(default_factory=list, max_length=20)
    notes: str | None = Field(default=None, max_length=2000)


class SubmissionReceipt(StrictModel):
    identifier: str
    status: Literal["recorded_unverified"]
    message: str


class ChatMessage(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class ChatContext(StrictModel):
    state_code: str | None = Field(default=None, min_length=2, max_length=2)
    planting_class: str | None = Field(default=None, max_length=80)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)


class ChatRequest(StrictModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=12)
    context: ChatContext = Field(default_factory=ChatContext)

    @model_validator(mode="after")
    def ends_with_user(self) -> "ChatRequest":
        if self.messages[-1].role != "user":
            raise ValueError("The final chat message must be from the user.")
        return self


class ChatToolUse(StrictModel):
    name: str


class ChatResponse(StrictModel):
    message: str
    model: str
    tools_used: list[ChatToolUse]
    recommendation: RecommendationResponse | None = None
    claim_boundary: str


class ChatStatusResponse(StrictModel):
    available: bool
    model: str
    role: str
