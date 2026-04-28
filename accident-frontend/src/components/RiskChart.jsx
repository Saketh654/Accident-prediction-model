import { useRef, useEffect } from "react";
import { Chart, registerables } from "chart.js";
Chart.register(...registerables);

export default function RiskChart({ scores, fps = 10 }) {
  const canvasRef = useRef(null);
  const chartRef  = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !scores?.length) return;

    // Destroy previous instance
    if (chartRef.current) {
      chartRef.current.destroy();
      chartRef.current = null;
    }

    const ctx      = canvasRef.current.getContext("2d");
    const gradient = ctx.createLinearGradient(0, 0, 0, 220);
    gradient.addColorStop(0, "rgba(230, 51, 35, 0.18)");
    gradient.addColorStop(1, "rgba(230, 51, 35, 0)");

    const labels = scores.map((_, i) => (i / fps).toFixed(1));

    // Inline plugin draws the dashed threshold line at 0.7
    const thresholdPlugin = {
      id: "thresholdLine",
      afterDraw(chart) {
        const { ctx: c, chartArea: { left, right }, scales: { y } } = chart;
        const yPos = y.getPixelForValue(0.7);
        c.save();
        c.strokeStyle = "rgba(230,51,35,0.4)";
        c.lineWidth   = 1;
        c.setLineDash([5, 4]);
        c.beginPath();
        c.moveTo(left, yPos);
        c.lineTo(right, yPos);
        c.stroke();
        c.fillStyle = "rgba(230,51,35,0.6)";
        c.font      = "10px 'IBM Plex Mono', monospace";
        c.fillText("alert threshold", left + 6, yPos - 5);
        c.restore();
      },
    };

    chartRef.current = new Chart(ctx, {
      type: "line",
      plugins: [thresholdPlugin],
      data: {
        labels,
        datasets: [{
          label:              "Risk Score",
          data:               scores,
          borderColor:        "#e63323",
          borderWidth:        2,
          backgroundColor:    gradient,
          pointRadius:        scores.length > 60 ? 0 : 3,
          pointBackgroundColor: "#e63323",
          tension:            0.35,
          fill:               true,
        }],
      },
      options: {
        responsive:          true,
        maintainAspectRatio: true,
        interaction:         { mode: "index", intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#0d0d0d",
            titleFont: { family: "'IBM Plex Mono', monospace", size: 11 },
            bodyFont:  { family: "'IBM Plex Mono', monospace", size: 12 },
            padding:   10,
            callbacks: {
              title: (items) => `t = ${items[0].label}s`,
              label: (item)  => ` risk: ${(item.raw * 100).toFixed(1)}%`,
            },
          },
        },
        scales: {
          x: {
            title: {
              display: true, text: "Time (s)",
              font:  { family: "'IBM Plex Mono', monospace", size: 10 },
              color: "rgba(13,13,13,0.45)",
            },
            grid:  { color: "rgba(13,13,13,0.06)" },
            ticks: {
              font:          { family: "'IBM Plex Mono', monospace", size: 10 },
              color:         "rgba(13,13,13,0.4)",
              maxTicksLimit: 10,
            },
          },
          y: {
            min: 0, max: 1,
            title: {
              display: true, text: "Risk",
              font:  { family: "'IBM Plex Mono', monospace", size: 10 },
              color: "rgba(13,13,13,0.45)",
            },
            grid:  { color: "rgba(13,13,13,0.06)" },
            ticks: {
              font:     { family: "'IBM Plex Mono', monospace", size: 10 },
              color:    "rgba(13,13,13,0.4)",
              callback: (v) => `${(v * 100).toFixed(0)}%`,
            },
          },
        },
      },
    });

    return () => {
      chartRef.current?.destroy();
      chartRef.current = null;
    };
  }, [scores, fps]);

  return <canvas ref={canvasRef} />;
}