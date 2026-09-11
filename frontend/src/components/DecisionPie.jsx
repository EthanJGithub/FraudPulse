import { ResponsivePie } from "@nivo/pie";

const COLORS = { ALLOW: "#6a669f", REVIEW: "#d7b66c", FLAG: "#c97b68" };
const theme = {
  text: { fill: "#66647f" },
  tooltip: { container: { background: "#ffffff", color: "#222038" } },
};

export default function DecisionPie({ counts, onSliceClick }) {
  const data = Object.keys(COLORS)
    .map((k) => ({ id: k, label: k, value: counts?.[k] || 0, color: COLORS[k] }))
    .filter((d) => d.value > 0);
  if (!data.length) return <p className="muted">No transactions yet.</p>;
  return (
    <div className="chart">
      <ResponsivePie
        data={data}
        margin={{ top: 24, right: 24, bottom: 40, left: 24 }}
        innerRadius={0.76} padAngle={1.2} cornerRadius={3}
        colors={(d) => d.data.color}
        borderWidth={1} borderColor={{ from: "color", modifiers: [["darker", 0.5]] }}
        arcLabelsTextColor="#0b0f17"
        arcLinkLabelsColor={{ from: "color" }} arcLinkLabelsTextColor="#66647f"
        onClick={onSliceClick ? (d) => onSliceClick(d.id) : undefined}
        theme={theme}
      />
    </div>
  );
}
