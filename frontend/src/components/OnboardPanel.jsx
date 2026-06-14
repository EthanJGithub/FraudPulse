import { useState } from "react";
import { api } from "../api.js";

// A tiny, realistic labelled CSV (different schema than ULB) to show the agent
// reasoning about arbitrary column roles — analyze-only, never retrains.
const EXAMPLE_CSV = `account_id,amount_usd,seconds_elapsed,feat_a,feat_b,feat_c,is_fraud
A1001,149.62,0,-1.36,-0.07,2.54,0
A1002,2.69,406,1.19,0.27,0.17,0
A1003,378.66,738,-1.16,0.88,-0.31,0
A1004,123.50,1120,0.40,-0.50,1.80,0
A1005,4099.99,1402,-3.04,2.10,-5.40,1
A1006,9.99,1899,0.10,0.05,0.22,0
A1007,1809.68,2300,-2.31,1.76,-4.10,1
A1008,67.88,2710,0.50,-0.20,0.90,0
A1009,15.99,3100,0.30,0.10,0.40,0
A1010,2500.00,3502,-2.90,1.95,-4.80,1`;

export default function OnboardPanel({ online }) {
  const [csv, setCsv] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const run = async (text) => {
    setBusy(true); setError(null); setResult(null);
    try {
      setResult(await api.onboardUpload(text));
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  };

  const onFile = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = () => setCsv(String(reader.result || ""));
    reader.readAsText(f);
  };

  const cfg = result?.config;
  return (
    <div className="card" style={{ marginTop: 16 }}>
      <h3>Dataset Onboarding Agent <span className="muted" style={{ fontWeight: 400 }}>· analyze-only</span></h3>
      <p className="muted" style={{ marginTop: -4 }}>
        Paste or upload a labelled transaction CSV. An LLM agent profiles it,
        reasons about which columns are the target / amount / time / features,
        self-corrects against the data, and returns its proposed schema. It does
        not retrain the shared demo model.
      </p>

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
        <button className="btn btn2" disabled={busy} onClick={() => { setCsv(EXAMPLE_CSV); run(EXAMPLE_CSV); }}>
          Try example
        </button>
        <label className="btn btn3" style={{ cursor: "pointer" }}>
          Upload CSV<input type="file" accept=".csv,text/csv" hidden onChange={onFile} />
        </label>
      </div>

      {!online && <p className="muted">⚠ Onboarding agent offline (no LLM configured on the server).</p>}
      {error && <div className="error">Onboarding error: {error}</div>}

      {result && (
        <div className="onboard-result">
          <div className="grid3">
            <div className="kpi"><div className="k">Provider</div><div className="v">{result.provider || result.model}</div></div>
            <div className="kpi"><div className="k">Attempts</div><div className="v">{result.attempts}</div></div>
            <div className="kpi"><div className="k">Validation</div>
              <div className={`v ${result.validation?.ok ? "green" : "red"}`}>
                {result.validation?.ok ? "✓ valid" : "✗ failed"}
              </div></div>
          </div>
          {cfg && (
            <table className="data" style={{ marginTop: 12 }}>
              <tbody>
                <tr><td className="muted">Dataset</td><td>{cfg.name}</td></tr>
                <tr><td className="muted">Target column</td><td><code>{cfg.target_col}</code> (fraud = {String(cfg.positive_label)})</td></tr>
                <tr><td className="muted">Amount column</td><td><code>{cfg.amount_col ?? "—"}</code></td></tr>
                <tr><td className="muted">Time column</td><td><code>{cfg.time_col ?? "—"}</code></td></tr>
                <tr><td className="muted">Feature columns</td><td>{cfg.feature_cols?.length ?? 0}</td></tr>
              </tbody>
            </table>
          )}
          {result.reasoning && (
            <div className="reasoning">
              <div className="k">Agent reasoning</div>
              <p>{result.reasoning}</p>
            </div>
          )}
          {result.notice && <p className="muted">{result.notice}</p>}
        </div>
      )}
    </div>
  );
}
