import { useState } from "react";
import { api } from "../api.js";

// Four labelled example datasets with deliberately different characteristics,
// served as static files (each ≥200 rows so none trip the small-sample advisory)
// and fetched on demand — see loadAndRun. The agent's insight for each is
// generated live by the LLM (not hard-coded); these only supply the raw CSV.
const EXAMPLES = {
  valid: {
    label: "Clean — well-formed, suitable",
    url: "/examples/clean.csv",
  },
  flagged: {
    label: "Flagged — id/leakage risk",
    url: "/examples/flagged.csv",
  },
  mixed: {
    label: "Mixed — workable, missing-value gaps",
    url: "/examples/mixed.csv",
  },
  medium_fraud: {
    label: "Elevated fraud rate — likely resampled (warning)",
    url: "/examples/medium_fraud.csv",
  },
  high_fraud: {
    label: "Implausible fraud rate — near-balanced, likely leakage",
    url: "/examples/high_fraud.csv",
  },
  production: {
    label: "Production-scale — 1,000 rows",
    url: "/examples/production_scale.csv",
  },
};

// Severity → all-caps tag for the flag warnings (backend severities: high /
// medium / info). Border + tag colour are driven by the matching .qflag.<sev> CSS.
const SEV_TAG = { high: "CRITICAL", medium: "WARNING", info: "ADVISORY" };

// DATA QUALITY (structural verdict) → risk tier pill.
const TIER = {
  ready: { label: "TIER 1 — Production Ready", cls: "green" },
  review: { label: "TIER 2 — Needs Review", cls: "amber" },
  flagged: { label: "TIER 3 — Flagged", cls: "red" },
};

const VENDOR = { groq: "Groq", anthropic: "Anthropic", openai: "OpenAI", ollama: "Ollama" };

// Format a raw provider string ("groq:llama-3.1-8b-instant") into a clean
// "Vendor / Family" engine label without echoing the full model slug.
function formatEngine(provider, model) {
  const raw = String(provider || model || "");
  const [vendorRaw, modelRaw = ""] = raw.split(":");
  const vendor = VENDOR[vendorRaw?.toLowerCase()] ||
    (vendorRaw ? vendorRaw.charAt(0).toUpperCase() + vendorRaw.slice(1) : "LLM");
  let fam = modelRaw;
  const m = modelRaw.match(/llama[-_]?(\d+(?:\.\d+)?)/i);
  if (m) fam = `LLaMA ${m[1]}`;
  else if (modelRaw) fam = modelRaw.split(/[-_]/).slice(0, 2).join(" ");
  return fam ? `${vendor} / ${fam}` : vendor;
}

// PRODUCTION READINESS — operational fitness derived from flag severities.
// Distinct from DATA QUALITY (structural): a clean-but-tiny dataset can be
// TIER 1 yet still need validation before deploy.
function productionReadiness(reasons) {
  const sevs = new Set((reasons || []).map((r) => r.severity));
  if (sevs.has("high") || sevs.has("medium")) return { label: "Not Production Ready", cls: "red" };
  if (sevs.has("info")) return { label: "Validate Before Deploy", cls: "amber" };
  return { label: "Production Ready", cls: "green" };
}

export default function OnboardPanel({ online }) {
  const [csv, setCsv] = useState("");
  const [example, setExample] = useState("valid");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showReasoning, setShowReasoning] = useState(false);

  const run = async (text) => {
    setBusy(true); setError(null); setResult(null); setShowReasoning(false);
    try {
      setResult(await api.onboardUpload(text));
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  };

  const loadAndRun = async () => {
    const ex = EXAMPLES[example];
    let text = ex.csv;
    if (!text && ex.url) {
      // Static example served from /public — fetched on demand (no silent fallback:
      // a fetch failure surfaces as an explicit error rather than an empty run).
      try {
        const resp = await fetch(ex.url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        text = await resp.text();
      } catch (e) {
        setError(`Could not load "${ex.label}": ${e.message}`);
        return;
      }
    }
    setCsv(text);
    run(text);
  };

  const onFile = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = () => setCsv(String(reader.result || ""));
    reader.readAsText(f);
  };

  const cfg = result?.config;
  const facts = result?.dataset_facts;
  const quality = result?.quality;
  const tier = quality && (TIER[quality.verdict] || { label: quality.verdict, cls: "" });
  const prod = quality && productionReadiness(quality.reasons);
  return (
    <div className="card" style={{ marginTop: 16 }}>
      <h3>Dataset Onboarding Agent <span className="muted" style={{ fontWeight: 400 }}>· analyze-only</span></h3>
      <p className="muted" style={{ marginTop: -4 }}>
        Paste or upload a labelled transaction CSV — or try a built-in example. An
        LLM agent profiles it, reasons about which columns are the target / amount /
        time / features, self-corrects against the data, and returns its proposed
        schema plus a plain-language read on what the data actually is. It does not
        retrain the shared demo model.
      </p>

      <div className="example-row">
        <select className="example-select" value={example} onChange={(e) => setExample(e.target.value)}>
          {Object.entries(EXAMPLES).map(([k, v]) => (
            <option key={k} value={k}>{v.label}</option>
          ))}
        </select>
        <button className="btn btn2" disabled={busy || !online} onClick={loadAndRun}>
          {busy ? "Analyzing…" : "Load & analyze example"}
        </button>
      </div>

      <textarea
        className="csvbox" value={csv} spellCheck={false}
        placeholder="col_a,col_b,is_fraud&#10;1.0,2.0,0&#10;..."
        onChange={(e) => setCsv(e.target.value)}
      />

      <div className="btnrow">
        <button className="btn" disabled={busy || !csv.trim() || !online}
          onClick={() => run(csv)}>
          {busy ? "Analyzing…" : "Analyze with agent"}
        </button>
        <label className="btn btn3" style={{ cursor: "pointer" }}>
          Upload CSV<input type="file" accept=".csv,text/csv" hidden onChange={onFile} />
        </label>
      </div>

      {!online && <p className="muted">Onboarding agent offline (no LLM configured on the server).</p>}
      {error && <div className="error">Onboarding error: {error}</div>}

      {result && (
        <div className="onboard-result">
          {result.insight && (
            <div className="insight">
              <div className="k">Agent insight — what this data is</div>
              <p>{result.insight}</p>
            </div>
          )}

          <div className="grid3" style={{ marginTop: 14 }}>
            <div className="kpi">
              <div className="k">Data quality</div>
              <div className="tier-pill-wrap">
                <span className={`tier-pill ${tier?.cls || ""}`}>{tier?.label || "—"}</span>
              </div>
              <div className="k" style={{ marginTop: 14 }}>Production readiness</div>
              <div className="tier-pill-wrap">
                <span className={`tier-pill ${prod?.cls || ""}`}>{prod?.label || "—"}</span>
              </div>
            </div>
            <div className="kpi"><div className="k">Fraud rate</div>
              <div className="v" style={{ fontSize: "1.3rem" }}>{facts?.fraud_rate_pct != null ? `${facts.fraud_rate_pct}%` : "—"}</div></div>
            <div className="kpi">
              <div className="k">AI Engine —</div>
              <div className="v engine-name">{formatEngine(result.provider, result.model)}</div>
              <div className={`engine-sub ${result.validation?.ok ? "ok" : "bad"}`}>
                {result.validation?.ok ? "Schema inference verified" : "Schema inference failed"}
              </div>
            </div>
          </div>

          {!!quality?.reasons?.length && (
            <ul className="quality-flags">
              {quality.reasons.map((r, i) => (
                <li key={i} className={`qflag ${r.severity}`}>
                  <span className="sev-tag">{SEV_TAG[r.severity] || String(r.severity).toUpperCase()}</span>
                  <span className="qflag-text">{r.text}</span>
                </li>
              ))}
            </ul>
          )}

          {cfg && (
            <table className="data" style={{ marginTop: 12 }}>
              <tbody>
                <tr><td className="muted">Dataset</td><td>{cfg.name}</td></tr>
                <tr><td className="muted">Rows / fraud rows</td><td>{facts?.n_rows ?? "—"} / {facts?.n_fraud_rows ?? "—"}</td></tr>
                <tr><td className="muted">Target column</td><td><code>{cfg.target_col}</code> (fraud = {String(cfg.positive_label)})</td></tr>
                <tr><td className="muted">Amount / Time</td><td><code>{cfg.amount_col ?? "—"}</code> / <code>{cfg.time_col ?? "—"}</code></td></tr>
                <tr><td className="muted">Feature columns</td><td>{cfg.feature_cols?.length ?? 0}</td></tr>
                {!!facts?.suspected_id_columns_excluded?.length && (
                  <tr><td className="muted">Suspected id cols</td><td className="warn">{facts.suspected_id_columns_excluded.join(", ")}</td></tr>
                )}
                {!!facts?.high_missingness_columns?.length && (
                  <tr><td className="muted">High missingness</td><td className="warn">{facts.high_missingness_columns.join(", ")}</td></tr>
                )}
              </tbody>
            </table>
          )}
          {result.reasoning && (
            <div className="reasoning collapsible">
              <button
                type="button"
                className="reasoning-head"
                aria-expanded={showReasoning}
                onClick={() => setShowReasoning((v) => !v)}
              >
                <span className="k">Schema-mapping rationale</span>
                <span className={`chev ${showReasoning ? "open" : ""}`} aria-hidden="true">›</span>
              </button>
              <div className={`reasoning-body ${showReasoning ? "open" : ""}`}>
                <p>{result.reasoning}</p>
              </div>
            </div>
          )}
          {result.notice && <p className="muted">{result.notice}</p>}
        </div>
      )}
    </div>
  );
}
