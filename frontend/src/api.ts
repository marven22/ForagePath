import type {
  ApiErrorBody,
  BoundaryCollection,
  GridCell,
  HealthResponse,
  PlantingClassSummary,
  RecommendationResponse,
  StateSummary,
  SubmissionReceipt,
  ChatMessage,
  ChatContext,
  ChatResponse,
  ChatStatusResponse,
  LatestResearchValidation,
} from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, { headers: { Accept: "application/json" } });
  } catch {
    throw new ApiError("The recommendation service is not reachable. Please try again shortly.", 0);
  }
  if (!response.ok) {
    let body: ApiErrorBody = {};
    try {
      body = (await response.json()) as ApiErrorBody;
    } catch {
      // Preserve the generic message when an upstream service does not return JSON.
    }
    throw new ApiError(body.message || `The service returned an error (${response.status}).`, response.status);
  }
  return response.json() as Promise<T>;
}

async function submit<T>(path: string, payload: object): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new ApiError("The feedback service is not reachable. Please try again shortly.", 0);
  }
  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as ApiErrorBody;
    throw new ApiError(body.message || `The service returned an error (${response.status}).`, response.status);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthResponse>("/health"),
  states: async () => (await request<{ states: StateSummary[] }>("/v1/states")).states,
  plantingClasses: async (state: string) =>
    (await request<{ planting_classes: PlantingClassSummary[] }>(
      `/v1/states/${encodeURIComponent(state)}/planting-classes`,
    )).planting_classes,
  cells: async (state: string) =>
    (await request<{ cells: GridCell[] }>(`/v1/states/${encodeURIComponent(state)}/cells`)).cells,
  boundaries: (state: string) =>
    request<BoundaryCollection>(`/v1/states/${encodeURIComponent(state)}/boundaries`),
  latestResearchValidation: () => request<LatestResearchValidation>("/v1/research/validation/latest"),
  chatStatus: () => request<ChatStatusResponse>("/v1/chat/status"),
  chat: (messages: ChatMessage[], context: ChatContext) =>
    submit<ChatResponse>("/v1/chat", { messages, context }),
  recommendation: (parameters: {
    state: string;
    plantingClass: string;
    latitude?: number;
    longitude?: number;
    cellId?: string;
  }) => {
    const search = new URLSearchParams({
      state: parameters.state,
      planting_class: parameters.plantingClass,
    });
    if (parameters.cellId) search.set("cell_id", parameters.cellId);
    if (parameters.latitude !== undefined) search.set("latitude", String(parameters.latitude));
    if (parameters.longitude !== undefined) search.set("longitude", String(parameters.longitude));
    return request<RecommendationResponse>(`/v1/recommendations?${search.toString()}`);
  },
  feedback: (payload: {
    release_id: string; state_code: string; cell_id: string; planting_class: string;
    selected_plants: string[]; helpful_rating: number;
    intended_action: "considering" | "planning" | "planted" | "not_using"; comments?: string;
  }) => submit<SubmissionReceipt>("/v1/feedback", payload),
  outcome: (payload: {
    feedback_id: string; observation_date: string; plants_attempted: string[];
    establishment_fraction?: number; flowering_observed?: boolean; pollinators_observed?: boolean;
    observed_groups?: string[]; notes?: string;
  }) => submit<SubmissionReceipt>("/v1/outcomes", payload),
};
