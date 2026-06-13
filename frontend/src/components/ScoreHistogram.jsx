import { ResponsiveBar } from "@nivo/bar";

const theme = {
  text: { fill: "#8b98b3", fontSize: 10 },
  axis: { ticks: { text: { fill: "#8b98b3" } }, legend: { text: { fill: "#8b98b3" } } },
  grid: { line: { stroke: "#263149" } },
  tooltip: { container: { background: "#1a2335", color: "#e8edf6" } },
};

export default function ScoreHistogram({ histogram }) {
  if (!histogram || !histogram.length) return <p className="muted">No data yet.</p>;
  // Log-ish emphasis: legit transactions dominate bin 0, so color by risk band.
  const data = histogram.map((h, i) => ({
    bin: `${i * 5}`,
    count: h.count,
    color: i >= 18 ? "#ff4d5e" : i >= 1 ? "#ffb020" : "#28a745",
  }));
  return (
    <div className="chart">
      <ResponsiveBar
        data={data} keys={["count"]} indexBy="bin"
        margin={{ top: 16, right: 16, bottom: 46, left: 56 }} padding={0.15}
        colors={(b) => b.data.color} theme={theme}
        axisBottom={{ tickValues: data.filter((_, i) => i % 4 === 0).map((d) => d.bin),
                      legend: "fraud probability (%)", legendPosition: "middle", legendOffset: 38 }}
        axisLeft={{ legend: "transactions", legendPosition: "middle", legendOffset: -46, tickSize: 0 }}
        enableLabel={false}
      />
    </div>
  );
}
