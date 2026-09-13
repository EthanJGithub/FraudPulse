import { useEffect, useRef, useState } from "react";
import { api } from "./api.js";
import DecisionPie from "./components/DecisionPie.jsx";
import ScoreHistogram from "./components/ScoreHistogram.jsx";
import OnboardPanel from "./components/OnboardPanel.jsx";
import SystemHealth from "./components/SystemHealth.jsx";

function Kpi({ k, v, cls }) {
  return <div className="kpi"><div className="k">{k}</div><div className={`v ${cls || ""}`}>{v}</div></div>;
}

const FILTERS = ["ALL", "ALLOW", "REVIEW", "FLAG"];

export default function App() {
  const [health, setHealth] = useState(null);
  const [meta, setMeta] = useState(null);
  const [stats, setStats] = useState(null);
  const [system, setSystem] = useState(null);
  const [feed, setFeed] = useState([]);
  const [filter, setFilter] = useState("ALL");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState(null);
  const streamRef = useRef(null);
  const refreshing = useRef(false);
  const bufRef = useRef([]);
  const filterRef = useRef(filter);
  filterRef.current = filter;

  const refresh = async () => {
    if (refreshing.current) return;
    refreshing.current = true;
    try {
      const f = filterRef.current;
      const [s, t, sys, serviceHealth] = await Promise.all([
        api.stats(),
        api.transactions(15, f === "ALL" ? undefined : f),
        api.system().catch(() => null),
        api.health().catch(() => null),
      ]);
      setStats(s); setFeed(t.transactions); if (sys) setSystem(sys); setError(null);
      if (serviceHealth) { setHealth(serviceHealth); api.modelInfo().then(setMeta).catch(() => {}); }
    } catch (e) { setError(e.message); setHealth(null); }
    finally { refreshing.current = false; }
  };

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth({ status: "error" }));
    api.modelInfo().then(setMeta).catch(() => {});
    refresh();
    const id = setInterval(refresh, 3000);   // live polling
    return () => clearInterval(id);
  }, []);

  // Re-pull the feed immediately when the decision filter changes.
  useEffect(() => { refresh(); /* eslint-disable-next-line */ }, [filter]);

  // Live stream: replay a random REAL transaction (from the committed sample,
  // via /sample) every ~1.4s while enabled — varied real amounts and a
  // realistic fraud/legit mix, no hardcoded vectors. Refills a buffer of 20 to
  // keep request volume low.
  useEffect(() => {
    if (!streaming) return;
    bufRef.current = [];
    streamRef.current = setInterval(async () => {
      try {
        if (bufRef.current.length === 0) {
          const r = await api.sample(20);
          bufRef.current = r.transactions || [];
        }
        const t = bufRef.current.shift();
        if (!t) return;
        await api.score({ txn_id: `live-${Date.now()}`, Amount: t.Amount, Time: t.Time, features: t.features });
        refresh();
      } catch (e) { setError(e.message); }
    }, 1400);
    return () => clearInterval(streamRef.current);
  }, [streaming]);

  const dc = stats?.decision_counts || {};
  const conf = stats?.confusion;
  const precision = conf ? conf.tp / Math.max(conf.tp + conf.fp, 1) : null;
  const recall = conf ? conf.tp / Math.max(conf.tp + conf.fn, 1) : null;

  return (
    <div className="app">
      <div className="header">
        <div>
          <div className="brand">Fraud<span>Pulse</span></div>
          
        </div>
        <nav className="product-nav" aria-label="Project links"><a href="https://ethanjgithub.github.io/">Portfolio ↗</a><a href="https://github.com/EthanJGithub/FraudPulse" target="_blank" rel="noopener">Source code ↗</a></nav>
        <div className="status">
          <div className="srow"><span className="dot" /><span>{health?.status === "ok" ? "Service connected" : "Connecting…"}</span></div>
          <div className="srow">
            <span className={`sdot ${health?.status === "ok" ? "ok" : "bad"}`} /><span>API</span>
            <span className="sdiv">·</span>
            <span className={`sdot ${health?.models_loaded ? "ok" : "bad"}`} /><span>models</span>
          </div>
          {meta && <div className="srow smetric">PR-AUC {meta.pr_auc} · ROC-AUC {meta.roc_auc}</div>}
        </div>
      </div>

      <section className="workspace-intro"><div><span className="eyebrow">Risk operations / overview</span><h1>Every transaction. A clearer signal.</h1><p>Monitor fraud risk, investigate model decisions, and make sense of incoming data.</p></div><span className="workspace-tag">XGBOOST + ISOLATIONFOREST</span></section>
      {health?.status !== "ok" && <div className="card" role="status"><h3>Your example is ready</h3><p>The live scoring service is starting. Explore a recorded model run immediately, or wait for the live controls.</p><a className="btn" href="https://ethanjgithub.github.io/demos/fraudpulse.html">Explore recorded results ↗</a></div>}
      <p className="muted">Dashboard metrics describe the replay stream, which contains an enriched selection of fraud cases. They are separate from final-test model metrics.</p>
      {error && <div className="error" role="status">Connecting to the scoring service. Free-tier startup may take a moment. {error}</div>}

      <div className="grid4">
        <Kpi k="Transactions Scored" v={(stats?.total ?? 0).toLocaleString()} />
        <Kpi k="Active Alerts" v={(stats?.alerts ?? 0).toLocaleString()} cls="amber" />
        <Kpi k="Flagged (blocked)" v={(dc.FLAG ?? 0).toLocaleString()} cls="red" />
        <Kpi k="Amount at Risk" v={`$${(stats?.amount_at_risk ?? 0).toLocaleString()}`} cls="red" />
      </div>

      <div className="btnrow">
        <button className={`btn ${streaming ? "btn2" : ""}`} disabled={health?.status !== "ok"} onClick={() => setStreaming((s) => !s)}>
          {streaming ? "⏸ Stop live stream" : "▶ Start live transaction stream"}
        </button>
        <span className="muted" style={{ alignSelf: "center" }}>
          Replays real transactions through the scoring API; alerts update live.
        </span>
      </div>

      <div className="grid2">
        <div className="card">
          <h3>Decision Mix</h3>
          <DecisionPie counts={dc} onSliceClick={(id) => setFilter(id)} />
        </div>
        <div className="card">
          <h3>Fraud-Probability Distribution {filter !== "ALL" && <span className="muted" style={{ fontWeight: 400 }}>· {filter}</span>}</h3>
          <ScoreHistogram histogram={stats?.histograms?.[filter] ?? stats?.score_histogram} />
        </div>
      </div>

      {conf && (
        <div className="grid3" style={{ marginTop: 16 }}>
          <Kpi k="Replay Precision" v={`${(precision * 100).toFixed(1)}%`} cls="green" />
          <Kpi k="Replay Recall" v={`${(recall * 100).toFixed(1)}%`} cls="green" />
          <Kpi k="Confusion (TP·FP·FN·TN)" v={`${conf.tp}·${conf.fp}·${conf.fn}·${conf.tn}`} />
        </div>
      )}

      <div className="card" style={{ marginTop: 16 }}>
        <div className="feed-head">
          <h3>Live Decision Feed</h3>
          <div className="filters">
            {FILTERS.map((f) => (
              <button key={f} className={`fbtn ${f} ${filter === f ? "active" : ""}`}
                onClick={() => setFilter(f)}>
                {f === "ALL" ? "All" : f}
              </button>
            ))}
          </div>
        </div>
        <table className="data">
          <thead><tr><th>Txn</th><th>Amount</th><th>Fraud Prob</th><th>Anomaly</th><th>Decision</th></tr></thead>
          <tbody>
            {feed.map((a) => (
              <tr key={a.txn_id}>
                <td className="scoreline">{a.txn_id}</td>
                <td>${a.amount?.toLocaleString()}</td>
                <td>{(a.fraud_probability * 100).toFixed(1)}%</td>
                <td>{a.anomaly_score?.toFixed(2)}{a.is_anomaly ? " ⚠" : ""}</td>
                <td><span className={`pill ${a.decision}`}>{a.decision}</span></td>
              </tr>
            ))}
            {!feed.length && <tr><td colSpan="5" className="muted">
              {filter === "ALL" ? "No transactions yet — start the live stream." : `No ${filter} transactions yet.`}
            </td></tr>}
          </tbody>
        </table>
      </div>

      <SystemHealth system={system} />

      <OnboardPanel online={health?.onboarding_agent_online} />
      <footer className="product-foot"><span>FraudPulse · Built by Ethan Jones</span><span>Real transactions. Explainable decisions.</span></footer>
    </div>
  );
}
