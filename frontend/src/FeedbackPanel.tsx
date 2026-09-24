import { FormEvent, useState } from "react";
import { api, ApiError } from "./api";
import type { RecommendationResponse } from "./types";

export default function FeedbackPanel({ result }: { result: RecommendationResponse }) {
  const [rating, setRating] = useState(0);
  const [action, setAction] = useState<"considering" | "planning" | "planted" | "not_using">("considering");
  const [comments, setComments] = useState("");
  const [feedbackId, setFeedbackId] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [showOutcome, setShowOutcome] = useState(false);
  const [observationDate, setObservationDate] = useState(new Date().toISOString().slice(0, 10));
  const [establishment, setEstablishment] = useState("75");
  const [flowering, setFlowering] = useState(false);
  const [pollinators, setPollinators] = useState(false);
  const [outcomeNotes, setOutcomeNotes] = useState("");

  const saveFeedback = async (event: FormEvent) => {
    event.preventDefault();
    if (!rating) { setMessage("Choose a helpfulness rating first."); return; }
    setBusy(true); setMessage(null);
    try {
      const receipt = await api.feedback({
        release_id: result.release.release_id,
        state_code: result.request.state_code,
        cell_id: result.request.resolved_cell_id,
        planting_class: result.request.planting_class,
        selected_plants: result.recommendation.plants.map((plant) => plant.scientific_name),
        helpful_rating: rating, intended_action: action, comments: comments || undefined,
      });
      setFeedbackId(receipt.identifier); setMessage("Thank you. Your feedback was recorded separately from verified evidence.");
    } catch (reason) { setMessage(reason instanceof ApiError ? reason.message : "Feedback could not be saved."); }
    finally { setBusy(false); }
  };

  const saveOutcome = async (event: FormEvent) => {
    event.preventDefault();
    if (!feedbackId) return;
    setBusy(true); setMessage(null);
    try {
      await api.outcome({
        feedback_id: feedbackId, observation_date: observationDate,
        plants_attempted: result.recommendation.plants.map((plant) => plant.scientific_name),
        establishment_fraction: Number(establishment) / 100,
        flowering_observed: flowering, pollinators_observed: pollinators,
        notes: outcomeNotes || undefined,
      });
      setMessage("Outcome recorded as unverified farmer-reported evidence."); setShowOutcome(false);
    } catch (reason) { setMessage(reason instanceof ApiError ? reason.message : "Outcome could not be saved."); }
    finally { setBusy(false); }
  };

  return (
    <article className="feedback-card">
      <div><p className="eyebrow">Improve the system</p><h3>Was this recommendation useful?</h3><p>No name, email, or exact farm coordinate is collected.</p></div>
      {!feedbackId ? (
        <form onSubmit={saveFeedback}>
          <div className="rating" aria-label="Helpfulness rating">
            {[1, 2, 3, 4, 5].map((value) => <button type="button" aria-label={`${value} out of 5`} className={rating >= value ? "active" : ""} onClick={() => setRating(value)} key={value}>★</button>)}
          </div>
          <label><span>What will you do?</span><select value={action} onChange={(event) => setAction(event.target.value as typeof action)}><option value="considering">Still considering it</option><option value="planning">Planning to plant</option><option value="planted">Already planted</option><option value="not_using">Will not use it</option></select></label>
          <label><span>Comments (optional)</span><textarea maxLength={1500} value={comments} onChange={(event) => setComments(event.target.value)} placeholder="What helped, or what was missing?" /></label>
          <button className="secondary-button" disabled={busy}>{busy ? "Saving…" : "Submit feedback"}</button>
        </form>
      ) : (
        <div className="feedback-receipt"><p>{message}</p><button className="secondary-button" onClick={() => setShowOutcome(!showOutcome)}>{showOutcome ? "Close outcome form" : "Record a planting outcome"}</button></div>
      )}
      {feedbackId && showOutcome && (
        <form className="outcome-form" onSubmit={saveOutcome}>
          <h4>What happened after planting?</h4>
          <div className="outcome-grid">
            <label><span>Observation date</span><input type="date" required value={observationDate} onChange={(event) => setObservationDate(event.target.value)} /></label>
            <label><span>Plants established</span><select value={establishment} onChange={(event) => setEstablishment(event.target.value)}><option value="0">None</option><option value="25">About 25%</option><option value="50">About 50%</option><option value="75">About 75%</option><option value="100">All</option></select></label>
          </div>
          <div className="check-row"><label><input type="checkbox" checked={flowering} onChange={(event) => setFlowering(event.target.checked)} /> Flowering observed</label><label><input type="checkbox" checked={pollinators} onChange={(event) => setPollinators(event.target.checked)} /> Pollinators observed</label></div>
          <label><span>Outcome notes (optional)</span><textarea maxLength={2000} value={outcomeNotes} onChange={(event) => setOutcomeNotes(event.target.value)} /></label>
          <button className="secondary-button" disabled={busy}>{busy ? "Saving…" : "Save outcome"}</button>
        </form>
      )}
      {message && !feedbackId && <p className="form-message" role="status">{message}</p>}
    </article>
  );
}
