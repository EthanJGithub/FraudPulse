import { useState } from "react";
import { api } from "../api.js";

// Three labelled example datasets with deliberately different characteristics.
// The agent's insight for each is generated live by the LLM (not hard-coded) —
// these only supply the raw CSV the agent reasons about.
const EXAMPLES = {
  valid: {
    label: "✅ Clean — well-formed, suitable",
    csv: `amount,elapsed_sec,pc1,pc2,pc3,is_fraud
12.40,0,-0.21,0.44,0.10,0
88.10,210,0.15,-0.32,0.22,0
4200.00,455,-3.10,2.40,-4.90,1
33.75,690,0.05,0.18,-0.11,0
9.99,940,0.22,-0.07,0.31,0
156.30,1180,-0.44,0.51,0.02,0
2750.50,1395,-2.85,1.92,-4.10,1
61.20,1620,0.10,-0.21,0.17,0
24.00,1870,0.31,0.09,-0.05,0
410.90,2090,-0.62,0.73,-0.40,0
18.45,2330,0.27,-0.14,0.20,0
3890.00,2560,-3.30,2.55,-5.10,1
72.60,2800,0.08,0.12,-0.09,0
145.00,3040,-0.38,0.40,0.05,0
6.50,3290,0.35,-0.02,0.28,0
220.75,3510,-0.51,0.60,-0.22,0
49.90,3760,0.12,-0.18,0.19,0
99.99,3990,0.02,0.21,-0.07,0`,
  },
  flagged: {
    label: "🚩 Flagged — tiny + id/leakage risk",
    csv: `user_account_id,amt,t,feat_x,is_fraud
U-90431,120.50,0,0.31,0
U-90432,18.20,300,0.12,0
U-90433,5400.00,610,-4.20,1
U-90434,42.75,905,0.08,0
U-90435,9.99,1240,0.25,0
U-90436,77.40,1560,-0.11,0
U-90437,33.10,1880,0.19,0`,
  },
  mixed: {
    label: "🟡 Mixed — workable, missing time + gaps",
    csv: `amount,sensor_a,sensor_b,sensor_c,label
14.20,0.31,,0.10,0
210.50,-0.44,0.51,0.02,0
3300.00,-3.10,2.40,,1
56.75,0.05,,-0.11,0
9.40,0.22,-0.07,0.31,0
410.30,-0.62,0.73,-0.40,0
2890.00,-2.85,,-4.10,1
61.20,0.10,-0.21,0.17,0
24.00,0.31,0.09,,0
133.90,-0.38,0.40,0.05,0
18.45,0.27,,0.20,0
2750.00,-3.30,2.55,-5.10,1
72.60,0.08,0.12,-0.09,0
99.99,0.02,,-0.07,0
145.00,-0.51,0.60,-0.22,0
49.90,0.12,-0.18,0.19,0`,
  },
};

export default function OnboardPanel({ online }) {
  const [csv, setCsv] = useState("");
  const [example, setExample] = useState("valid");
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

  const loadAndRun = () => {
    const text = EXAMPLES[example].csv;
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

      {!online && <p className="muted">⚠ Onboarding agent offline (no LLM configured on the server).</p>}
      {error && <div className="error">Onboarding error: {error}</div>}

      {result && (
        <div className="onboard-result">
          {result.insight && (
            <div className="insight">
              <div className="k">🧠 Agent insight — what this data is</div>
              <p>{result.insight}</p>
            </div>
          )}

          <div className="grid3" style={{ marginTop: 14 }}>
            <div className="kpi"><div className="k">Provider</div><div className="v" style={{ fontSize: "1rem" }}>{result.provider || result.model}</div></div>
            <div className="kpi"><div className="k">Fraud rate</div>
              <div className="v" style={{ fontSize: "1.3rem" }}>{facts?.fraud_rate_pct != null ? `${facts.fraud_rate_pct}%` : "—"}</div></div>
            <div className="kpi"><div className="k">Validation</div>
              <div className={`v ${result.validation?.ok ? "green" : "red"}`} style={{ fontSize: "1.3rem" }}>
                {result.validation?.ok ? "✓ valid" : "✗ failed"}
              </div></div>
          </div>

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
            <div className="reasoning">
              <div className="k">Schema-mapping rationale</div>
              <p>{result.reasoning}</p>
            </div>
          )}
          {result.notice && <p className="muted">{result.notice}</p>}
        </div>
      )}
    </div>
  );
}
