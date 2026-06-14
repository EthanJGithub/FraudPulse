const BASE = import.meta.env.VITE_API_BASE || "/api/v1";

async function req(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  return res.json();
}

export const api = {
  health: () => req("/health"),
  modelInfo: () => req("/model/info"),
  score: (txn) => req("/score", { method: "POST", body: JSON.stringify(txn) }),
  stats: () => req("/monitoring/stats"),
  alerts: (limit = 25) => req(`/monitoring/alerts?limit=${limit}`),
  sample: (n = 20) => req(`/sample?n=${n}`),
};
