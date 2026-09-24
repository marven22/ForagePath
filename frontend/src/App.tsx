import { FormEvent, useEffect, useRef, useState } from "react";
import { api, ApiError } from "./api";
import FeedbackPanel from "./FeedbackPanel";
import MapPicker from "./MapPicker";
import ChatAgent from "./ChatAgent";
import type {
  BoundaryCollection,
  GridCell,
  HealthResponse,
  PlantingClassSummary,
  RecommendationResponse,
  RecommendedPlant,
  LatestResearchValidation,
  StateSummary,
} from "./types";

const STATE_CENTERS: Record<string, [number, number]> = {
  PA: [40.88, -77.8],
  MD: [39.05, -76.64],
  MI: [44.31, -85.6],
};

const MONTHS = [
  ["Jan", 1, 4], ["Feb", 5, 8], ["Mar", 9, 13], ["Apr", 14, 17],
  ["May", 18, 21], ["Jun", 22, 26], ["Jul", 27, 30], ["Aug", 31, 35],
  ["Sep", 36, 39], ["Oct", 40, 43], ["Nov", 44, 48], ["Dec", 49, 52],
] as const;

function percent(value: number | null | undefined): string {
  return value == null ? "Not available" : `${Math.round(value * 100)}%`;
}

function relativeScore(value: number | null | undefined): string {
  return value == null ? "Not available" : `${value.toFixed(2)} / 1`;
}

function readable(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function stateCoverageLabel(state: StateSummary): string {
  if (state.guidance_level === "localized_with_statewide_fallback") return `${state.published_cells} local grid areas + statewide fallback`;
  if (state.guidance_level === "statewide_only") return "statewide guidance";
  if (state.guidance_level === "unavailable") return "evidence still insufficient";
  return `${state.published_cells}/${state.modeled_cells} published grid areas`;
}

function LeafMark() {
  return (
    <svg aria-hidden="true" viewBox="0 0 48 48" className="brand-mark">
      <path d="M40 8C23 8 11 17 11 31c0 5 3 9 8 9 14 0 23-12 21-32Z" fill="currentColor" opacity=".22" />
      <path d="M8 41c7-11 15-18 27-26M21 29c1-5 1-9 0-13M22 28c5 0 9 1 13 3" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" />
      <circle cx="10" cy="15" r="3" fill="#e8b44b" />
      <path d="m10 9 1.4 3.1 3.4.4-2.5 2.3.7 3.4-3-1.7-3 1.7.7-3.4-2.5-2.3 3.4-.4L10 9Z" fill="#e8b44b" opacity=".7" />
    </svg>
  );
}

function Icon({ name }: { name: "pin" | "bee" | "calendar" | "shield" | "arrow" | "target" }) {
  const paths = {
    pin: <><path d="M12 21s7-5.1 7-12a7 7 0 1 0-14 0c0 6.9 7 12 7 12Z" /><circle cx="12" cy="9" r="2.5" /></>,
    bee: <><ellipse cx="12" cy="13" rx="4.2" ry="6" /><path d="M8.3 10.5h7.4M8.2 14h7.6M10 7 8 4M14 7l2-3M8 11C4 7 2 10 4 13M16 11c4-4 6-1 4 2" /></>,
    calendar: <><rect x="3" y="5" width="18" height="16" rx="2" /><path d="M7 3v4M17 3v4M3 10h18M7 14h3M14 14h3M7 18h3" /></>,
    shield: <path d="M12 3 4.5 6v5.2c0 4.6 3 8.3 7.5 9.8 4.5-1.5 7.5-5.2 7.5-9.8V6L12 3Zm-3 9 2 2 4-4" />,
    arrow: <><path d="M5 12h14M14 7l5 5-5 5" /></>,
    target: <><circle cx="12" cy="12" r="8" /><circle cx="12" cy="12" r="3" /><path d="m14 10 6-6" /></>,
  };
  return <svg aria-hidden="true" className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{paths[name]}</svg>;
}

function FloweringTimeline({ plants }: { plants: RecommendedPlant[] }) {
  return (
    <div className="timeline-wrap">
      <div className="month-row" aria-hidden="true">
        <span />
        <div className="month-grid">
          {MONTHS.map(([month, start, end]) => (
            <span key={month} style={{ gridColumn: `${start} / ${end + 1}` }}>{month}</span>
          ))}
        </div>
      </div>
      {plants.map((plant) => {
        const active = new Set(plant.modeled_weeks_at_or_above_20_percent);
        return (
          <div className="timeline-row" key={plant.scientific_name}>
            <span className="timeline-label"><i>{plant.scientific_name}</i></span>
            <div className="week-grid" aria-label={`Modeled flowering weeks for ${plant.scientific_name}`}>
              {Array.from({ length: 52 }, (_, index) => index + 1).map((week) => (
                <span
                  key={week}
                  className={`${active.has(week) ? "active" : ""} ${plant.modeled_peak_flowering_week === week ? "peak" : ""}`}
                  title={`Week ${week}${plant.modeled_peak_flowering_week === week ? " — modeled peak" : active.has(week) ? " — modeled flowering probability ≥20%" : ""}`}
                />
              ))}
            </div>
          </div>
        );
      })}
      <div className="timeline-legend">
        <span><i className="legend-swatch active" /> Modeled probability ≥20%</span>
        <span><i className="legend-swatch peak" /> Modeled peak week</span>
      </div>
    </div>
  );
}

function PlantCard({ plant, index }: { plant: RecommendedPlant; index: number }) {
  const groups = plant.documented_native_bee_groups.map((entry) => readable(entry.bee_group));
  return (
    <article className="plant-card">
      <div className="plant-number">{String(index + 1).padStart(2, "0")}</div>
      <div className="plant-card-content">
        {plant.image?.local_path && <img className="plant-image" src={plant.image.local_path} alt={`${plant.common_name || plant.scientific_name}`} loading="lazy" />}
        <p className="eyebrow">{plant.growth_form || plant.coarse_habitat || "Plant profile"}</p>
        <h3>{plant.common_name || plant.display_name}<small><i>{plant.scientific_name}</i></small></h3>
        <div className="plant-stats">
          <div><span>Relative suitability index</span><strong>{relativeScore(plant.local_suitability_score)}</strong></div>
          <div><span>Phenology confidence</span><strong>{percent(plant.phenology_confidence)}</strong></div>
          <div><span>Documented native bees</span><strong>{plant.documented_eligible_native_bee_species_count}</strong></div>
        </div>
        <div className="tag-list" aria-label="Documented native bee groups">
          {groups.slice(0, 4).map((group) => <span key={group}>{group}</span>)}
          {groups.length > 4 && <span>+{groups.length - 4} groups</span>}
        </div>
        <details className="plant-guidance">
          <summary>Planting considerations</summary>
          <p>{plant.moisture_guidance || "Moisture guidance unavailable"}</p>
          <p>{plant.light_guidance || "Light guidance unavailable"}</p>
          {plant.acidic_preference && <p>Coarse evidence indicates an acidic-soil preference.</p>}
          <small>{plant.practical_note}</small>
          {plant.image?.source_url && <a href={plant.image.source_url} target="_blank" rel="noreferrer">Photo: {plant.image.creator || "iNaturalist contributor"} · {plant.image.license}</a>}
        </details>
      </div>
    </article>
  );
}

function LoadingResult() {
  return (
    <section className="results-shell loading-shell" aria-live="polite" aria-label="Loading recommendation">
      <div className="spinner" />
      <div><h2>Evaluating the local evidence</h2><p>Matching flowering timing, suitability and documented native-bee relationships.</p></div>
    </section>
  );
}

function ResearchValidationCard({ research }: { research: LatestResearchValidation }) {
  return (
    <article className="research-card">
      <div className="research-card-header">
        <div><p className="eyebrow">Research transparency</p><h3>How this approach was tested</h3></div>
      </div>
      <p>This historical test is shown for transparency. It does not replace the evidence available for the location you selected.</p>
      <div className="research-grid">
        {research.views.map((view) => (
          <div key={view.design}>
            <h4>{view.label}</h4>
            {view.methods.filter((method) => method.method !== "random_mean").map((method) => (
              <div className="research-method" key={method.method}>
                <span>{method.label}</span>
                <strong>{percent(method.bee_week_coverage)} observed bee-week coverage</strong>
                {method.difference_vs_documented_links != null && <small>
                  {method.difference_vs_documented_links >= 0 ? "+" : ""}{percent(method.difference_vs_documented_links)} versus documented links only
                </small>}
              </div>
            ))}
          </div>
        ))}
      </div>
      <small className="research-limit">It does not establish bee health, abundance, pollination, or crop yield.</small>
    </article>
  );
}

function Results({ result, onAsk, research }: { result: RecommendationResponse; onAsk: () => void; research: LatestResearchValidation | null }) {
  const {
    recommendation, request, spatial_decision: spatial, local_evidence: evidence,
    interpretation, documented_native_bee_support: support,
  } = result;
  const totalSpecies = new Set(support.flatMap((group) => group.species)).size;
  const baselineRetained = !recommendation.robust_change_accepted;
  const evidenceNumber = (value: number | null) => value == null ? "Not available" : value.toLocaleString();
  return (
    <section className="results-shell" aria-labelledby="results-heading">
      <div className="results-header">
        <div>
          <p className="eyebrow">Your recommendation · {request.state_code}</p>
          <h2 id="results-heading">Your potential-forage mixture</h2>
          <p>
            {recommendation.mixture_size} native plants for a {request.planting_class}
            {spatial.county_name ? ` in ${spatial.county_name} County` : ""}.
          </p>
        </div>
        <div className="result-actions">
          <button className="ask-result-button" onClick={onAsk}><Icon name="bee" />Ask about this result</button>
          <button className="download-button" onClick={async () => {
            const { downloadRecommendationPdf } = await import("./report");
            downloadRecommendationPdf(result);
          }}>Download PDF</button>
          <div className={`decision-badge evidence-${spatial.evidence_grade}`}>
            <Icon name="shield" />
            <span><small>Location evidence</small>{spatial.label}</span>
          </div>
        </div>
      </div>

      <div className="result-metrics">
        <div><Icon name="pin" /><span><small>Recommendation source</small><strong>{spatial.level === "statewide" ? "Statewide consensus" : `${request.distance_to_modeled_cell_km.toFixed(1)} km away`}</strong></span></div>
        <div><Icon name="bee" /><span><small>Documented native bees</small><strong>{totalSpecies} species</strong></span></div>
        <div><Icon name="target" /><span><small>Guidance level</small><strong>{readable(spatial.level)}</strong></span></div>
        <div><Icon name="calendar" /><span><small>Years of flowering evidence</small><strong>{evidenceNumber(evidence.model_years)}</strong></span></div>
      </div>

      <article className={`spatial-card evidence-${spatial.evidence_grade}`}>
        <div>
          <p className="eyebrow">Why this spatial level was used</p>
          <h3>{spatial.label}</h3>
          <p>{spatial.reason}</p>
          {spatial.environmental_similarity != null && <p>
            Environmental similarity: <strong>{Math.round(spatial.environmental_similarity * 100)}%</strong>
            {` across ${spatial.environment_features_compared} measured features`}.
          </p>}
        </div>
        <div className="coverage-grid">
          <span><small>Plant occurrence records</small><strong>{evidenceNumber(evidence.plant_occurrence_records)}</strong></span>
          <span><small>Flowering observations</small><strong>{evidenceNumber(evidence.flowering_observations)}</strong></span>
          <span><small>Bee occurrence records</small><strong>{evidenceNumber(evidence.bee_occurrence_records)}</strong></span>
          <span><small>Bee sampling events</small><strong>{evidenceNumber(evidence.bee_sampling_events)}</strong></span>
          <span><small>Bee observation years</small><strong>{evidenceNumber(evidence.bee_observation_years)}</strong></span>
          <span><small>Cell validation</small><strong>{evidence.cell_validation_pass == null ? "Not available" : evidence.cell_validation_pass ? "Passed" : "Did not pass"}</strong></span>
        </div>
        <small className="coverage-note">Unavailable means the required cell-level audit data do not exist in this release; it does not mean zero observations.</small>
      </article>

      <div className="section-heading">
        <div><p className="eyebrow">Recommended composition</p><h3>Plants selected for this modeled area</h3></div>
        <p>Selection favors seasonal availability first, then uses documented bee relationships to refine among near-equivalent mixtures.</p>
      </div>
      <div className="plant-grid">
        {recommendation.plants.map((plant, index) => <PlantCard key={plant.scientific_name} plant={plant} index={index} />)}
      </div>

      <div className="detail-grid">
        <article className="detail-card flowering-card">
          <div className="card-heading"><Icon name="calendar" /><div><p className="eyebrow">Seasonal view</p><h3>Modeled flowering windows</h3></div></div>
          <p className="card-intro">These are model probabilities—not guarantees that every plant will flower during every highlighted week.</p>
          <FloweringTimeline plants={recommendation.plants} />
        </article>

        <article className="detail-card bee-card">
          <div className="card-heading"><Icon name="bee" /><div><p className="eyebrow">Pollinator scorecard</p><h3>Documented native-bee support</h3></div></div>
          <p className="card-intro">Only eligible native-bee species with documented links are shown here.</p>
          <div className="bee-groups">
            {support.map((group) => (
              <details key={group.bee_group}>
                <summary><span>{readable(group.bee_group)}</span><strong>{group.species_count}</strong></summary>
                <div className="species-list">{group.species.map((species) => <i key={species}>{species}</i>)}</div>
              </details>
            ))}
          </div>
        </article>
      </div>

      <article className="method-card">
        <div className="method-title"><span>How this decision was made</span></div>
        <div className="method-steps">
          <div><span>1</span><p><strong>Resolve location</strong>Try exact grid, county, environmental region and state, in that order.</p></div>
          <div><span>2</span><p><strong>Availability first</strong>Flowering timing and suitability establish the candidate set.</p></div>
          <div><span>3</span><p><strong>Evidence and uncertainty</strong>Documented native-bee links refine choices; robustness gates can retain the baseline.</p></div>
        </div>
        <div className="ai-note">
          <Icon name="shield" />
          <p><strong>{baselineRetained ? "The conservative mixture was retained." : "A robust alternative passed."}</strong> {recommendation.ai_gate_reason}</p>
        </div>
      </article>

      {research && <ResearchValidationCard research={research} />}

      <aside className="limits-card">
        <strong>What this result means</strong>
        <p>{interpretation.claim_boundary}</p>
        <p>{interpretation.important_note}</p>
      </aside>
      {spatial.level !== "statewide" && <FeedbackPanel result={result} />}
    </section>
  );
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [research, setResearch] = useState<LatestResearchValidation | null>(null);
  const [states, setStates] = useState<StateSummary[]>([]);
  const [classes, setClasses] = useState<PlantingClassSummary[]>([]);
  const [cells, setCells] = useState<GridCell[]>([]);
  const [boundaries, setBoundaries] = useState<BoundaryCollection | null>(null);
  const [state, setState] = useState("PA");
  const [plantingClass, setPlantingClass] = useState("herbaceous strip");
  const [mode, setMode] = useState<"map" | "coordinates">("map");
  const [latitude, setLatitude] = useState("40.00");
  const [longitude, setLongitude] = useState("-75.30");
  const [loading, setLoading] = useState(false);
  const [locating, setLocating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RecommendationResponse | null>(null);
  const [chatOpenRequest, setChatOpenRequest] = useState(0);
  const resultsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    Promise.all([api.health(), api.states()])
      .then(([healthResult, stateResults]) => { setHealth(healthResult); setStates(stateResults); })
      .catch((reason: ApiError) => setError(reason.message));
    api.latestResearchValidation().then(setResearch).catch(() => {
      // Research artifacts are supplementary; the validated recommendation service remains usable without them.
    });
  }, []);

  useEffect(() => {
    setResult(null);
    setError(null);
    const center = STATE_CENTERS[state];
    if (center) { setLatitude(center[0].toFixed(2)); setLongitude(center[1].toFixed(2)); }
    Promise.all([api.plantingClasses(state), api.cells(state), api.boundaries(state)])
      .then(([classResults, cellResults, boundaryResults]) => {
        setClasses(classResults);
        setCells(cellResults);
        setBoundaries(boundaryResults);
        if (!center && cellResults.length) {
          const latitudeCenter = cellResults.reduce((sum, cell) => sum + cell.center_latitude, 0) / cellResults.length;
          const longitudeCenter = cellResults.reduce((sum, cell) => sum + cell.center_longitude, 0) / cellResults.length;
          setLatitude(latitudeCenter.toFixed(2)); setLongitude(longitudeCenter.toFixed(2));
        }
        if (!cellResults.length) setMode("coordinates");
        if (!classResults.some((item) => item.planting_class === plantingClass)) {
          setPlantingClass(classResults[0]?.planting_class || "");
        }
      })
      .catch((reason: ApiError) => setError(reason.message));
  }, [state]);

  const locate = () => {
    if (!navigator.geolocation) { setError("This browser does not provide location access."); return; }
    setLocating(true); setError(null);
    navigator.geolocation.getCurrentPosition(
      (position) => {
        setLatitude(position.coords.latitude.toFixed(5));
        setLongitude(position.coords.longitude.toFixed(5));
        setLocating(false);
      },
      () => { setError("Location permission was unavailable. Enter coordinates instead."); setLocating(false); },
      { enableHighAccuracy: false, timeout: 10000 },
    );
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!stateAvailable || !selectedClass) {
      setError("A responsible recommendation is not available for this state yet.");
      return;
    }
    setError(null); setLoading(true); setResult(null);
    try {
      const response = await api.recommendation({
        state,
        plantingClass,
        ...(hasLocalCandidates ? { latitude: Number(latitude), longitude: Number(longitude) } : {}),
      });
      setResult(response);
      window.setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 80);
    } catch (reason) {
      setError(reason instanceof ApiError ? reason.message : "The recommendation could not be generated.");
    } finally {
      setLoading(false);
    }
  };

  const selectedClass = classes.find((item) => item.planting_class === plantingClass);
  const selectedState = states.find((item) => item.state_code === state);
  const stateAvailable = Boolean(selectedState?.published_cells);
  const hasDetailedMap = cells.length > 0 && Boolean(boundaries?.features.length);
  const hasLocalCandidates = cells.length > 0;
  const mapCenter: [number, number] = STATE_CENTERS[state] || (cells.length
    ? [
      cells.reduce((sum, cell) => sum + cell.center_latitude, 0) / cells.length,
      cells.reduce((sum, cell) => sum + cell.center_longitude, 0) / cells.length,
    ]
    : [40, -77]);

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="ForagePath home"><LeafMark /><span>Forage<span>Path</span></span></a>
        <nav aria-label="Primary navigation"><a href="#recommend">Get a recommendation</a><a href="#method">How it works</a></nav>
        <div className={`service-status ${health ? "online" : ""}`}><i />{health ? "Service available" : "Connecting"}</div>
      </header>

      <main id="top">
        <section className="hero">
          <div className="hero-copy">
            <p className="eyebrow hero-eyebrow"><span /> Evidence-guided pollinator habitat</p>
            <h1>Plant for the weeks<br />that <em>matter.</em></h1>
            <p className="hero-lead">Build a native planting mixture around modeled flowering gaps, local suitability and documented native-bee relationships.</p>
            <div className="hero-notes"><span><Icon name="shield" />Uncertainty-aware</span><span><Icon name="bee" />Native-bee focused</span><span><Icon name="pin" />United States</span></div>
          </div>
          <div className="hero-visual" aria-hidden="true">
            <div className="orbit orbit-one" /><div className="orbit orbit-two" />
            <div className="visual-flower"><span /><span /><span /><span /><span /><i /></div>
            <div className="visual-bee"><span className="bee-body" /><span className="wing wing-one" /><span className="wing wing-two" /></div>
            <p>From scattered evidence<br /><strong>to one defensible mixture</strong></p>
          </div>
        </section>

        <section className="recommend-section" id="recommend">
          <div className="form-intro"><p className="eyebrow">Start here</p><h2>Describe the planting location</h2><p>{hasDetailedMap ? "We use the finest defensible level: exact grid, county, environmental region, then statewide guidance." : hasLocalCandidates ? "Enter coordinates to check whether your location falls in one of this state's replay-supported local grids. Other locations receive statewide guidance." : "This state currently receives statewide evidence guidance. Local grid guidance is still being expanded."}</p></div>
          <form className="recommend-form" onSubmit={submit}>
            <div className="field-row">
              <label><span>State</span><select value={state} onChange={(event) => setState(event.target.value)}>{states.map((item) => <option value={item.state_code} key={item.state_code}>{item.state_name} · {stateCoverageLabel(item)}</option>)}</select></label>
              <label><span>Planting type</span><select value={plantingClass} disabled={!classes.length} onChange={(event) => setPlantingClass(event.target.value)}>{classes.map((item) => <option value={item.planting_class} key={item.planting_class}>{readable(item.planting_class)}</option>)}</select></label>
              <div className="mixture-readout"><span>{hasLocalCandidates ? "Evidence-gated mixture size" : "Guidance mixture size"}</span><strong>{selectedClass?.mixture_size || "—"} <small>plants</small></strong></div>
            </div>

            {!stateAvailable && <div className="statewide-note"><strong>Recommendation not available yet.</strong><span>This state does not yet meet the evidence threshold for a responsible statewide mix.</span></div>}
            {stateAvailable && !hasDetailedMap && !hasLocalCandidates && <div className="statewide-note"><strong>Statewide guidance.</strong><span>This mix uses evidence gathered for the selected state. It is not a local grid-level recommendation.</span></div>}
            {stateAvailable && !hasDetailedMap && hasLocalCandidates && <div className="statewide-note"><strong>Localized coverage is limited.</strong><span>Enter coordinates to use a replay-supported local grid when one exists; otherwise the app returns statewide guidance.</span></div>}

            {hasDetailedMap && <fieldset>
              <legend>Location method</legend>
              <div className="segmented">
                <button type="button" className={mode === "map" ? "selected" : ""} onClick={() => setMode("map")}><Icon name="pin" /> Interactive map</button>
                <button type="button" className={mode === "coordinates" ? "selected" : ""} onClick={() => setMode("coordinates")}><Icon name="target" /> Enter coordinates</button>
              </div>
            </fieldset>}

            {hasLocalCandidates && (!hasDetailedMap || mode === "coordinates") ? (
              <div className="location-row">
                <label><span>Latitude</span><input type="number" min="-90" max="90" step="any" required value={latitude} onChange={(event) => setLatitude(event.target.value)} /></label>
                <label><span>Longitude</span><input type="number" min="-180" max="180" step="any" required value={longitude} onChange={(event) => setLongitude(event.target.value)} /></label>
                <button className="location-button" type="button" onClick={locate} disabled={locating}><Icon name="target" />{locating ? "Locating…" : "Use my location"}</button>
              </div>
            ) : hasDetailedMap ? <MapPicker cells={cells} boundaries={boundaries} center={mapCenter} selected={[Number(latitude), Number(longitude)]} onChoose={(lat, lon) => { setLatitude(lat.toFixed(5)); setLongitude(lon.toFixed(5)); }} /> : null}

            {error && <div className="error-message" role="alert"><strong>We could not complete that request.</strong><span>{error}</span></div>}
            <div className="form-action"><p><Icon name="shield" /> The system may retain a conservative baseline when local evidence is insufficient.</p><button className="primary-button" disabled={loading || !health || !stateAvailable}>{loading ? "Evaluating…" : "Build my mixture"}<Icon name="arrow" /></button></div>
          </form>
        </section>

        <ChatAgent stateCode={state} plantingClass={plantingClass}
          latitude={Number(latitude)} longitude={Number(longitude)}
          openRequest={chatOpenRequest} hasRecommendation={Boolean(result)}
          onRecommendation={(recommendation) => {
            setResult(recommendation);
            window.setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 80);
          }} />

        <div ref={resultsRef}>{loading && <LoadingResult />}{result && <Results result={result} research={research} onAsk={() => setChatOpenRequest((value) => value + 1)} />}</div>

        <section className="method-overview" id="method">
          <div><p className="eyebrow">Built for defensible decisions</p><h2>A recommendation should show its work.</h2></div>
          <p>ForagePath keeps observed evidence, model estimates and uncertainty separate. It can decline to change a baseline when the data do not support a confident local alternative.</p>
        </section>
      </main>

      <footer><div className="brand footer-brand"><LeafMark /><span>Forage<span>Path</span></span></div><p>Research prototype for potential forage continuity. Not a field-validated planting prescription.</p><span>Evidence-guided recommendations</span></footer>
    </div>
  );
}
