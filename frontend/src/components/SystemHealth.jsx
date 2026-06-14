// Operational observability: request latency, error rate, throughput, and PSI
// model-drift. This is the "monitoring system" view — production health, not
// just a model metric.

const STATUS_COLOR = {
  stable: "var(--accent)", moderate: "var(--amber)", significant: "var(--red)",
  "no-reference": "var(--muted)", "insufficient-data": "var(--muted)",
};

function Stat({ label, value, sub, cls }) {
  return (
    <div className="hstat">
      <div className="k">{label}</div>
      <div className={`hv ${cls || ""}`}>{value}</div>
      {sub && <div className="hsub">{sub}</div>}
    </div>
  );
}

export default function SystemHealth({ system }) {
  if (!system) return null;
  const m = system.metrics || {};
  const lat = m.latency_ms || {};
  const drift = system.drift || {};
  const errPct = ((m.error_rate ?? 0) * 100).toFixed(2);
  const errCls = (m.error_rate ?? 0) > 0.01 ? "red" : "green";
  const driftColor = STATUS_COLOR[drift.overall_status] || "var(--muted)";

  return (
    <div className="card" style={{ marginTop: 16 }}>
      <h3>System Health <span className="muted" style={{ fontWeight: 400 }}>· live observability</span></h3>

      <div className="hgrid">
        <Stat label="p50 latency" value={`${lat.p50 ?? 0} ms`} />
        <Stat label="p95 latency" value={`${lat.p95 ?? 0} ms`} cls={(lat.p95 ?? 0) > 750 ? "amber" : ""} />
        <Stat label="p99 latency" value={`${lat.p99 ?? 0} ms`} cls={(lat.p99 ?? 0) > 1500 ? "red" : ""} />
        <Stat label="Error rate" value={`${errPct}%`} cls={errCls} sub={`${m.errors_total ?? 0} / ${m.requests_total ?? 0} reqs`} />
        <Stat label="Throughput" value={`${m.throughput_rpm ?? 0}/min`} />
        <Stat label="Uptime" value={fmtUptime(m.uptime_seconds)} />
      </div>

      <div className="drift-block">
        <div className="drift-head">
          <span className="k">Model drift (PSI)</span>
          <span className="drift-badge" style={{ color: driftColor, borderColor: driftColor }}>
            {drift.overall_status} · max {drift.max_psi ?? 0}
          </span>
        </div>
        {drift.reference_available ? (
          <table className="data">
            <thead><tr><th>Signal</th><th>PSI</th><th>Status</th></tr></thead>
            <tbody>
              {(drift.features || []).map((f) => (
                <tr key={f.feature}>
                  <td className="scoreline">{f.feature}</td>
                  <td>{f.psi}</td>
                  <td><span style={{ color: STATUS_COLOR[f.status] }}>{f.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="muted">No drift reference available.</p>
        )}
        <p className="muted" style={{ marginTop: 8 }}>
          PSI compares recent transactions ({drift.n_current ?? 0} scored) to the
          training distribution. &lt;0.10 stable · 0.10–0.25 investigate · &gt;0.25 retrain.
        </p>
      </div>
    </div>
  );
}

function fmtUptime(s) {
  if (!s) return "0s";
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}
