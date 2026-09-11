import { ResponsiveBar } from "@nivo/bar";

const theme = {
  text: { fill: "#66647f", fontSize: 10 },
  axis: { ticks: { text: { fill: "#66647f" } }, legend: { text: { fill: "#66647f" } } },
  grid: { line: { stroke: "#dedde9" } },
  tooltip: { container: { background: "#ffffff", color: "#222038" } },
};

export default function ScoreHistogram({ histogram }) {
  if (!histogram || !histogram.length) return <p className="muted">No data yet.</p>;
  // Log-ish emphasis: legit transactions dominate bin 0, so color by risk band.
  const data = histogram.map((h, i) => ({
    bin: `${i * 5}`,
    count: h.count,
    color: i >= 18 ? "#c97b68" : i >= 1 ? "#d7b66c" : "#6a669f",
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
