import { FormEvent, useEffect, useRef, useState } from "react";
import { api, ApiError } from "./api";
import type { ChatMessage, RecommendationResponse } from "./types";

interface Props {
  stateCode: string;
  plantingClass: string;
  latitude: number;
  longitude: number;
  openRequest: number;
  hasRecommendation: boolean;
  onRecommendation: (result: RecommendationResponse) => void;
}

const GREETING: ChatMessage = {
  role: "assistant",
  content: "I can explain the science, interpret your current result, or run the validated recommendation engine for the selected map location.",
};

function cleanMessage(content: string) {
  return content.replace(/\*\*/g, "").replace(/^#{1,6}\s+/gm, "");
}

function BeeIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><ellipse cx="12" cy="13" rx="4" ry="6" /><path d="M8.4 10.5h7.2M8.2 14h7.6M10 7 8 4M14 7l2-3M8 11C4 7 2 10 4 13M16 11c4-4 6-1 4 2" /></svg>;
}

export default function ChatAgent({
  stateCode, plantingClass, latitude, longitude, openRequest, hasRecommendation, onRecommendation,
}: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([GREETING]);
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(false);
  const [available, setAvailable] = useState<boolean | null>(null);
  const [model, setModel] = useState("");
  const [open, setOpen] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let active = true;
    api.chatStatus().then((status) => {
      if (active) { setAvailable(status.available); setModel(status.model); }
    }).catch(() => { if (active) setAvailable(false); });
    return () => { active = false; };
  }, []);

  useEffect(() => { if (openRequest > 0) setOpen(true); }, [openRequest]);

  useEffect(() => {
    if (!open) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const closeOnEscape = (event: KeyboardEvent) => { if (event.key === "Escape") setOpen(false); };
    window.addEventListener("keydown", closeOnEscape);
    window.setTimeout(() => inputRef.current?.focus(), 120);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [open]);

  useEffect(() => {
    if (open) endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, open]);

  const sendMessage = async (content: string) => {
    content = content.trim();
    if (!content || loading || !available) return;
    const next = [...messages, { role: "user" as const, content }].slice(-12);
    setMessages(next); setDraft(""); setLoading(true);
    try {
      const response = await api.chat(next, {
        state_code: stateCode, planting_class: plantingClass, latitude, longitude,
      });
      setMessages((current) => [...current, { role: "assistant" as const, content: response.message }].slice(-12));
      if (response.recommendation) {
        onRecommendation(response.recommendation);
      }
      setModel(response.model);
    } catch (reason) {
      const message = reason instanceof ApiError ? reason.message : "The assistant could not answer right now.";
      setMessages((current) => [...current, { role: "assistant" as const, content: message }].slice(-12));
    } finally { setLoading(false); }
  };
  const send = (event: FormEvent) => { event.preventDefault(); void sendMessage(draft); };

  return <>
    {!open && <button className="chat-launcher" onClick={() => setOpen(true)} aria-label="Open Ask ForagePath">
      <span><BeeIcon /></span><strong>Ask ForagePath</strong>
    </button>}

    {open && <div className="chat-overlay" onMouseDown={() => setOpen(false)}>
      <aside className="chat-drawer" role="dialog" aria-modal="true" aria-labelledby="chat-title"
        onMouseDown={(event) => event.stopPropagation()}>
        <header className="chat-heading">
          <div><p className="eyebrow">Evidence-grounded assistant</p><h2 id="chat-title">Ask ForagePath</h2></div>
          <button className="chat-close" onClick={() => setOpen(false)} aria-label="Close assistant">×</button>
        </header>
        <div className="chat-context">
          <span>{stateCode}</span><span>{plantingClass}</span>
          <small>{hasRecommendation ? "Current mixture available" : "Using selected map location"}</small>
        </div>
        <p className="chat-intro">It explains results and calls the same validated engine as the map. It cannot invent a mixture.</p>
        <div className="chat-messages" aria-live="polite">
          {messages.map((message, index) => <div className={`chat-message ${message.role}`} key={`${message.role}-${index}`}>
            <strong>{message.role === "assistant" ? "ForagePath" : "You"}</strong><p>{cleanMessage(message.content)}</p>
          </div>)}
          {loading && <div className="chat-message assistant"><strong>ForagePath</strong><p>Checking the evidence…</p></div>}
          <div ref={endRef} />
        </div>
        <div className="chat-suggestions">
          {hasRecommendation ? <>
            <button onClick={() => void sendMessage("Why were these plants selected for this location?")}>Why these plants?</button>
            <button onClick={() => void sendMessage("Which documented native bees does this mixture support?")}>Which native bees?</button>
            <button onClick={() => void sendMessage("How reliable is this recommendation?")}>How reliable is it?</button>
          </> : <>
            <button onClick={() => void sendMessage("Recommend a mixture for the selected location.")}>Recommend here</button>
            <button onClick={() => void sendMessage("How does this system make a recommendation?")}>How does it work?</button>
            <button onClick={() => void sendMessage("What can and cannot be claimed from these results?")}>What are the limits?</button>
          </>}
        </div>
        <form className="chat-form" onSubmit={send}>
          <input ref={inputRef} value={draft} maxLength={2000} onChange={(event) => setDraft(event.target.value)}
            placeholder={available === false ? "Chat is not configured" : "Ask about this location or result…"}
            disabled={!available || loading} aria-label="Message the ForagePath assistant" />
          <button disabled={!available || loading || !draft.trim()}>Send</button>
        </form>
        <div className="chat-footer">
          <span className={available ? "agent-ready" : "agent-offline"}>{available === null ? "Checking…" : available ? "Ready" : "Unavailable"}</span>
          <span>{model || "Grounded model"}</span><button onClick={() => setMessages([GREETING])}>Clear</button>
        </div>
      </aside>
    </div>}
  </>;
}
