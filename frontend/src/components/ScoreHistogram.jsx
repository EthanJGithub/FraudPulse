import { ResponsiveBar } from "@nivo/bar";

const theme = {
  text: { fill: "#687b6c", fontSize: 10 },
  axis: { ticks: { text: { fill: "#687b6c" } }, legend: { text: { fill: "#687b6c" } } },
  grid: { line: { stroke: "#e3e9dd" } },
  tooltip: { container: { background: "#ffffff", color: "#20382d" } },
};

export default function ScoreHistogram({ histogram }) {
  if (!histogram || !histogram.length) return <p className="muted">No data yet.</p>;
  // Log-ish emphasis: legit transactions dominate bin 0, so color by risk band.
  const data = histogram.map((h, i) => ({
    bin: `${i * 5}`,
    count: h.count,
    color: i >= 18 ? "#c97b68" : i >= 1 ? "#d7b66c" : "#789f66",
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
