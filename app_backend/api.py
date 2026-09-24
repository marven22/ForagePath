from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from .chat_agent import AgentServiceError, AgentUnavailableError, GroundedChatAgent
from .database import DEFAULT_DB
from .feedback_store import FeedbackStore
from .recommendation_service import RecommendationService
from .research_validation import latest_validation
from .validated_pipeline import catalog as validated_pipeline_catalog, state_policy as validated_state_policy
from .schemas import (
    GridCellListResponse,
    ChatRequest,
    ChatResponse,
    ChatStatusResponse,
    FeedbackSubmission,
    HealthResponse,
    PlantingClassListResponse,
    RecommendationResponse,
    OutcomeSubmission,
    ServiceIndex,
    StateListResponse,
    SubmissionReceipt,
)


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("pollinator_api")
API_VERSION = "1.2.0"


def database_path() -> Path:
    return Path(os.getenv("POLLINATOR_DATABASE_PATH", str(DEFAULT_DB))).resolve()


def allowed_origins() -> list[str]:
    raw = os.getenv(
        "POLLINATOR_ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://localhost:4173,"
        "http://127.0.0.1:3000,http://127.0.0.1:5173,http://127.0.0.1:4173",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = RecommendationService(database_path())
    health = service.health()
    if health["status"] != "ready":
        raise RuntimeError("The application database has no published recommendation release.")
    app.state.recommendations = service
    app.state.chat_agent = GroundedChatAgent(service)
    app.state.feedback = FeedbackStore()
    app.state.feedback.initialize()
    app.state.release = health["current_release"]
    LOGGER.info("Loaded release %s from the validated application database", app.state.release["release_id"])
    try:
        yield
    finally:
        app.state.feedback.engine.dispose()


app = FastAPI(
    title="Pollinator Forage Recommendation API",
    description=(
        "Versioned access to evidence-gated potential-forage recommendations. "
        "Outputs do not prove bee health, crop yield, establishment, or causal field benefit."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Accept", "Content-Type", "X-Request-ID"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))[:128]
    request.state.request_id = request_id
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "public, max-age=300" if request.method == "GET" else "no-store"
    LOGGER.info(
        "request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
        request_id, request.method, request.url.path, response.status_code,
        1000 * (time.perf_counter() - started),
    )
    return response


@app.exception_handler(LookupError)
async def not_found(request: Request, exc: LookupError):
    return JSONResponse(
        status_code=404,
        content={"error": "not_found", "message": str(exc), "request_id": request.state.request_id},
    )


@app.exception_handler(ValueError)
async def invalid_request(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=422,
        content={"error": "invalid_request", "message": str(exc), "request_id": request.state.request_id},
    )


def service(request: Request) -> RecommendationService:
    return request.app.state.recommendations


def feedback_store(request: Request) -> FeedbackStore:
    return request.app.state.feedback


def chat_agent(request: Request) -> GroundedChatAgent:
    return request.app.state.chat_agent


@app.get("/", response_model=ServiceIndex, tags=["service"])
def index() -> ServiceIndex:
    return ServiceIndex(
        service="Pollinator Forage Recommendation API",
        version=API_VERSION,
        documentation="/docs",
        health="/health",
    )


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health(request: Request) -> HealthResponse:
    result = service(request).health()
    if result["status"] != "ready" or not result["current_release"]:
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    return HealthResponse(status="ready", release=result["current_release"])


@app.get("/v1/states", response_model=StateListResponse, tags=["catalog"])
def states(request: Request) -> StateListResponse:
    return StateListResponse(states=service(request).list_states())


@app.get(
    "/v1/states/{state_code}/planting-classes",
    response_model=PlantingClassListResponse,
    tags=["catalog"],
)
def planting_classes(state_code: str, request: Request) -> PlantingClassListResponse:
    state = state_code.upper()
    return PlantingClassListResponse(
        state_code=state,
        planting_classes=service(request).list_planting_classes(state),
    )


@app.get("/v1/states/{state_code}/cells", response_model=GridCellListResponse, tags=["catalog"])
def cells(state_code: str, request: Request) -> GridCellListResponse:
    state = state_code.upper()
    return GridCellListResponse(state_code=state, cells=service(request).list_cells(state))


@app.get("/v1/states/{state_code}/boundaries", tags=["catalog"])
def boundaries(state_code: str, request: Request) -> JSONResponse:
    return JSONResponse(content=service(request).boundary(state_code))


@app.get("/v1/recommendations", response_model=RecommendationResponse, tags=["recommendations"])
def recommendation(
    request: Request,
    state: str = Query(min_length=2, max_length=2, description="Two-letter state code"),
    planting_class: str = Query(description="Currently herbaceous strip or woody hedgerow"),
    cell_id: str | None = Query(default=None),
    latitude: float | None = Query(default=None, ge=-90, le=90),
    longitude: float | None = Query(default=None, ge=-180, le=180),
    mixture_size: int | None = Query(default=None, ge=1, le=20),
) -> RecommendationResponse:
    payload = service(request).recommend(
        state_code=state,
        planting_class=planting_class,
        cell_id=cell_id,
        latitude=latitude,
        longitude=longitude,
        mixture_size=mixture_size,
    )
    return RecommendationResponse.model_validate(payload)


@app.get("/v1/research/validation/latest", tags=["research"])
def latest_research_validation() -> dict:
    """Expose the latest completed backtest without presenting it as a live recommendation."""
    return latest_validation()


@app.get("/v1/pipeline/states", tags=["research"])
def pipeline_states() -> dict:
    """Evidence-tiered national deployment plan; this is not a live recommendation endpoint."""
    return validated_pipeline_catalog()


@app.get("/v1/pipeline/states/{state_code}", tags=["research"])
def pipeline_state(state_code: str) -> dict:
    return validated_state_policy(state_code)


@app.get("/v1/chat/status", response_model=ChatStatusResponse, tags=["chat"])
def chat_status(request: Request) -> ChatStatusResponse:
    return ChatStatusResponse.model_validate(chat_agent(request).status())


@app.post("/v1/chat", response_model=ChatResponse, tags=["chat"])
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    try:
        result = chat_agent(request).chat(
            [item.model_dump() for item in payload.messages],
            payload.context.model_dump(exclude_none=True),
        )
        return ChatResponse.model_validate(result)
    except AgentUnavailableError as exc:
        return JSONResponse(status_code=503, content={
            "error": "chat_unavailable", "message": str(exc), "request_id": request.state.request_id,
        })
    except AgentServiceError:
        return JSONResponse(status_code=502, content={
            "error": "language_service_error",
            "message": "The chat assistant could not answer right now. The recommendation engine is still available.",
            "request_id": request.state.request_id,
        })


@app.post("/v1/feedback", response_model=SubmissionReceipt, status_code=201, tags=["feedback"])
def submit_feedback(payload: FeedbackSubmission, request: Request) -> SubmissionReceipt:
    # Validate that feedback refers to the exact published recommendation rather than arbitrary plants.
    expected = service(request).recommend(
        state_code=payload.state_code, planting_class=payload.planting_class, cell_id=payload.cell_id,
    )
    if expected["release"]["release_id"] != payload.release_id:
        raise ValueError("Feedback must refer to the current published release.")
    expected_plants = {item["scientific_name"] for item in expected["recommendation"]["plants"]}
    if {item.casefold() for item in payload.selected_plants} != expected_plants:
        raise ValueError("The submitted plants do not match the published recommendation.")
    identifier = feedback_store(request).save_feedback(payload.model_dump())
    return SubmissionReceipt(
        identifier=identifier, status="recorded_unverified",
        message="Feedback was stored separately from verified scientific evidence.",
    )


@app.post("/v1/outcomes", response_model=SubmissionReceipt, status_code=201, tags=["feedback"])
def submit_outcome(payload: OutcomeSubmission, request: Request) -> SubmissionReceipt:
    identifier = feedback_store(request).save_outcome(payload.model_dump())
    return SubmissionReceipt(
        identifier=identifier, status="recorded_unverified",
        message="The outcome was recorded as unverified farmer-reported evidence.",
    )
