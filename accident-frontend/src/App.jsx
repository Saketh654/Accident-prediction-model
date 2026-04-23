import { useState, useRef, useCallback, useEffect } from "react";
import axios from "axios";
import { Chart, registerables } from "chart.js";
Chart.register(...registerables);

// ─── Design tokens ────────────────────────────────────────────────────────────
const css = `
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --ink:      #0d0d0d;
    --paper:    #f5f2eb;
    --cream:    #ede9de;
    --danger:   #e63323;
    --warn:     #e8a020;
    --safe:     #2d9e6b;
    --rule:     rgba(13,13,13,0.12);
    --font-disp: 'Syne', sans-serif;
    --font-mono: 'IBM Plex Mono', monospace;
    --radius:   4px;
    --trans:    0.22s cubic-bezier(0.4,0,0.2,1);
  }

  html, body, #root { height: 100%; }

  body {
    background: var(--paper);
    color: var(--ink);
    font-family: var(--font-mono);
    -webkit-font-smoothing: antialiased;
  }

  /* ── Noise texture overlay ─────────────────────────────── */
  body::before {
    content: '';
    position: fixed; inset: 0; pointer-events: none; z-index: 9999;
    opacity: 0.028;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
    background-size: 200px 200px;
  }

  /* ── Shared layout ─────────────────────────────────────── */
  .page {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    animation: fadeIn 0.4s ease both;
  }

  @keyframes fadeIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:none; } }

  /* ── Header ────────────────────────────────────────────── */
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 40px;
    border-bottom: 1px solid var(--rule);
  }

  .logo {
    font-family: var(--font-disp);
    font-weight: 800;
    font-size: 18px;
    letter-spacing: -0.5px;
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .logo-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--danger);
    animation: pulse 2s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(0.85)} }

  .header-tag {
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    opacity: 0.4;
  }

  /* ── HOME PAGE ─────────────────────────────────────────── */
  .home-body {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 24px;
    gap: 48px;
  }

  .hero {
    text-align: center;
    max-width: 580px;
  }

  .hero-eyebrow {
    font-size: 11px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    opacity: 0.45;
    margin-bottom: 16px;
  }

  .hero-title {
    font-family: var(--font-disp);
    font-weight: 800;
    font-size: clamp(36px, 5vw, 62px);
    line-height: 1.05;
    letter-spacing: -2px;
    margin-bottom: 18px;
  }

  .hero-title span { color: var(--danger); }

  .hero-sub {
    font-size: 13px;
    line-height: 1.7;
    opacity: 0.55;
    max-width: 420px;
    margin: 0 auto;
  }

  /* ── Drop zone ─────────────────────────────────────────── */
  .dropzone-wrap { width: 100%; max-width: 540px; }

  .dropzone {
    border: 1.5px dashed rgba(13,13,13,0.25);
    border-radius: 8px;
    background: var(--cream);
    padding: 52px 32px;
    text-align: center;
    cursor: pointer;
    transition: border-color var(--trans), background var(--trans), transform var(--trans);
    position: relative;
    overflow: hidden;
  }

  .dropzone:hover, .dropzone.drag-over {
    border-color: var(--ink);
    background: #e8e4d8;
    transform: translateY(-2px);
  }

  .dropzone input[type="file"] {
    position: absolute; inset: 0; opacity: 0; cursor: pointer;
  }

  .dz-icon {
    font-size: 36px;
    margin-bottom: 14px;
    display: block;
    transition: transform var(--trans);
  }
  .dropzone:hover .dz-icon { transform: translateY(-4px); }

  .dz-primary {
    font-family: var(--font-disp);
    font-weight: 700;
    font-size: 16px;
    margin-bottom: 6px;
  }

  .dz-secondary {
    font-size: 11px;
    opacity: 0.45;
    letter-spacing: 0.5px;
  }

  .file-chosen {
    margin-top: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 16px;
    background: white;
    border: 1px solid var(--rule);
    border-radius: var(--radius);
    font-size: 12px;
  }

  .file-chosen-name {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-weight: 500;
  }

  .file-chosen-size { opacity: 0.4; }

  .file-clear {
    background: none; border: none; cursor: pointer;
    opacity: 0.35; font-size: 16px;
    transition: opacity var(--trans);
    padding: 0 4px;
  }
  .file-clear:hover { opacity: 0.8; }

  /* ── Analyze button ────────────────────────────────────── */
  .btn-analyze {
    margin-top: 20px;
    width: 100%;
    padding: 16px 32px;
    font-family: var(--font-disp);
    font-weight: 700;
    font-size: 15px;
    letter-spacing: 0.5px;
    background: var(--ink);
    color: var(--paper);
    border: none;
    border-radius: var(--radius);
    cursor: pointer;
    transition: background var(--trans), transform var(--trans), opacity var(--trans);
    position: relative;
    overflow: hidden;
  }

  .btn-analyze:hover:not(:disabled) {
    background: #222;
    transform: translateY(-1px);
  }

  .btn-analyze:disabled { opacity: 0.4; cursor: not-allowed; }

  /* loading shimmer */
  .btn-analyze.loading::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.12) 50%, transparent 100%);
    animation: shimmer 1.4s ease infinite;
  }
  @keyframes shimmer { from{transform:translateX(-100%)} to{transform:translateX(100%)} }

  /* ── Progress bar ──────────────────────────────────────── */
  .progress-wrap {
    margin-top: 16px;
    width: 100%;
    height: 3px;
    background: var(--rule);
    border-radius: 2px;
    overflow: hidden;
  }

  .progress-bar {
    height: 100%;
    background: var(--ink);
    border-radius: 2px;
    transition: width 0.3s ease;
  }

  .progress-label {
    margin-top: 8px;
    font-size: 11px;
    opacity: 0.5;
    letter-spacing: 0.5px;
  }

  /* ── RESULT PAGE ───────────────────────────────────────── */
  .result-body {
    flex: 1;
    padding: 32px 40px 60px;
    max-width: 1100px;
    margin: 0 auto;
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 36px;
  }

  .result-title {
    font-family: var(--font-disp);
    font-weight: 800;
    font-size: 28px;
    letter-spacing: -1px;
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .result-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-family: var(--font-mono);
  }

  .badge-alert { background: #fde8e6; color: var(--danger); }
  .badge-ok    { background: #e2f5ec; color: var(--safe); }

  /* ── Summary cards ─────────────────────────────────────── */
  .summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
  }

  .stat-card {
    background: var(--cream);
    border: 1px solid var(--rule);
    border-radius: 8px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
  }

  .stat-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    border-radius: 3px 0 0 3px;
  }

  .stat-card.danger::before { background: var(--danger); }
  .stat-card.warn::before   { background: var(--warn); }
  .stat-card.safe::before   { background: var(--safe); }
  .stat-card.neutral::before { background: var(--ink); }

  .stat-label {
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    opacity: 0.45;
    margin-bottom: 8px;
  }

  .stat-value {
    font-family: var(--font-disp);
    font-weight: 800;
    font-size: 28px;
    letter-spacing: -1px;
    line-height: 1;
  }

  .stat-value.danger { color: var(--danger); }
  .stat-value.warn   { color: var(--warn); }
  .stat-value.safe   { color: var(--safe); }

  /* ── Two-column content ────────────────────────────────── */
  .content-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
  }

  @media (max-width: 760px) {
    .content-grid { grid-template-columns: 1fr; }
    .result-body { padding: 24px 20px 48px; }
    .header { padding: 16px 20px; }
  }

  .panel {
    background: var(--cream);
    border: 1px solid var(--rule);
    border-radius: 8px;
    overflow: hidden;
  }

  .panel-header {
    padding: 14px 20px;
    border-bottom: 1px solid var(--rule);
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    opacity: 0.5;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .panel-body { padding: 20px; }

  /* ── Video player ──────────────────────────────────────── */
  .video-wrap {
    aspect-ratio: 16/9;
    background: #0d0d0d;
    border-radius: 0 0 8px 8px;
    overflow: hidden;
  }

  video {
    width: 100%; height: 100%;
    object-fit: contain;
    display: block;
  }

  /* ── Chart ─────────────────────────────────────────────── */
  .chart-panel { display: flex; flex-direction: column; }
  .chart-wrap  { padding: 16px 20px 20px; flex: 1; }

  /* ── Analysis text ─────────────────────────────────────── */
  .analysis-section {
    background: var(--cream);
    border: 1px solid var(--rule);
    border-radius: 8px;
    padding: 28px 32px;
  }

  .analysis-heading {
    font-family: var(--font-disp);
    font-weight: 700;
    font-size: 17px;
    margin-bottom: 12px;
  }

  .analysis-text {
    font-size: 13px;
    line-height: 1.75;
    opacity: 0.65;
    max-width: 720px;
  }

  .analysis-text strong { opacity: 1; color: var(--ink); font-weight: 500; }

  /* ── Back button ───────────────────────────────────────── */
  .btn-back {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: none;
    border: 1px solid var(--rule);
    padding: 9px 18px;
    border-radius: var(--radius);
    font-family: var(--font-mono);
    font-size: 12px;
    cursor: pointer;
    transition: border-color var(--trans), background var(--trans);
  }
  .btn-back:hover { border-color: var(--ink); background: white; }

  /* ── Error ─────────────────────────────────────────────── */
  .error-box {
    margin-top: 14px;
    padding: 12px 16px;
    border-left: 3px solid var(--danger);
    background: #fde8e6;
    border-radius: 0 var(--radius) var(--radius) 0;
    font-size: 12px;
    color: var(--danger);
  }

  /* ── Footer ────────────────────────────────────────────── */
  .footer {
    padding: 20px 40px;
    border-top: 1px solid var(--rule);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .footer-note {
    font-size: 11px;
    opacity: 0.3;
    letter-spacing: 0.3px;
  }
`;

// ─── Helpers ──────────────────────────────────────────────────────────────────
function fmtBytes(b) {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / (1024 * 1024)).toFixed(1)} MB`;
}

function riskLevel(v) {
  if (v >= 0.7) return { label: "HIGH", color: "danger", accent: "#e63323" };
  if (v >= 0.4) return { label: "MEDIUM", color: "warn", accent: "#e8a020" };
  return { label: "LOW", color: "safe", accent: "#2d9e6b" };
}

function computeSummary(scores) {
  if (!scores || scores.length === 0) return null;
  const max = Math.max(...scores);
  const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
  const alertFrames = scores.filter((s) => s >= 0.7).length;
  const firstAlert = scores.findIndex((s) => s >= 0.7);
  return { max, avg, alertFrames, firstAlert, total: scores.length };
}

// ─── Risk Chart component ─────────────────────────────────────────────────────
function RiskChart({ scores, fps = 10 }) {
  const canvasRef = useRef(null);
  const chartRef  = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !scores?.length) return;
    if (chartRef.current) { chartRef.current.destroy(); chartRef.current = null; }

    const ctx = canvasRef.current.getContext("2d");

    const gradient = ctx.createLinearGradient(0, 0, 0, 220);
    gradient.addColorStop(0, "rgba(230, 51, 35, 0.18)");
    gradient.addColorStop(1, "rgba(230, 51, 35, 0)");

    const labels = scores.map((_, i) => (i / fps).toFixed(1));

    // Threshold line drawn as inline plugin — fixes the "getter only" error
    const thresholdPlugin = {
      id: "thresholdLine",
      afterDraw(chart) {
        const { ctx: c, chartArea: { left, right }, scales: { y } } = chart;
        const yPos = y.getPixelForValue(0.7);
        c.save();
        c.strokeStyle = "rgba(230,51,35,0.4)";
        c.lineWidth = 1;
        c.setLineDash([5, 4]);
        c.beginPath();
        c.moveTo(left, yPos);
        c.lineTo(right, yPos);
        c.stroke();
        c.fillStyle = "rgba(230,51,35,0.6)";
        c.font = "10px 'IBM Plex Mono', monospace";
        c.fillText("alert threshold", left + 6, yPos - 5);
        c.restore();
      }
    };

    chartRef.current = new Chart(ctx, {
      type: "line",
      plugins: [thresholdPlugin],   // ← register here, not after creation
      data: {
        labels,
        datasets: [{
          label: "Risk Score",
          data: scores,
          borderColor: "#e63323",
          borderWidth: 2,
          backgroundColor: gradient,
          pointRadius: scores.length > 60 ? 0 : 3,
          pointBackgroundColor: "#e63323",
          tension: 0.35,
          fill: true,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#0d0d0d",
            titleFont: { family: "'IBM Plex Mono', monospace", size: 11 },
            bodyFont:  { family: "'IBM Plex Mono', monospace", size: 12 },
            padding: 10,
            callbacks: {
              title: (items) => `t = ${items[0].label}s`,
              label: (item)  => ` risk: ${(item.raw * 100).toFixed(1)}%`,
            }
          },
        },
        scales: {
          x: {
            title: {
              display: true, text: "Time (s)",
              font: { family: "'IBM Plex Mono', monospace", size: 10 },
              color: "rgba(13,13,13,0.45)"
            },
            grid: { color: "rgba(13,13,13,0.06)" },
            ticks: {
              font: { family: "'IBM Plex Mono', monospace", size: 10 },
              color: "rgba(13,13,13,0.4)",
              maxTicksLimit: 10
            },
          },
          y: {
            min: 0, max: 1,
            title: {
              display: true, text: "Risk",
              font: { family: "'IBM Plex Mono', monospace", size: 10 },
              color: "rgba(13,13,13,0.45)"
            },
            grid: { color: "rgba(13,13,13,0.06)" },
            ticks: {
              font: { family: "'IBM Plex Mono', monospace", size: 10 },
              color: "rgba(13,13,13,0.4)",
              callback: v => `${(v * 100).toFixed(0)}%`
            },
          }
        }
      }
    });

    return () => { chartRef.current?.destroy(); chartRef.current = null; };
  }, [scores, fps]);

  return <canvas ref={canvasRef} />;
}
// ─── Home Page ────────────────────────────────────────────────────────────────
function HomePage({ onResult }) {
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const inputRef = useRef(null);
  const [selectedModel, setSelectedModel] = useState("3dcnn");

  const handleFile = (f) => {
    if (!f) return;
    if (!f.type.startsWith("video/")) {
      setError("Please upload a video file.");
      return;
    }
    setFile(f);
    setError("");
  };

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files?.[0]);
  }, []);

  const onDragOver = (e) => {
    e.preventDefault();
    setDragging(true);
  };
  const onDragLeave = () => setDragging(false);

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setProgress(0);
    setError("");
    setStatus("Uploading video…");

    const fd = new FormData();
    fd.append("file", file);
    fd.append("model_name", selectedModel);

    try {
      setProgress(15);
      setStatus("Running inference…");

      const res = await axios.post("http://localhost:8000/analyze", fd, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (e) => {
          const pct = Math.round((e.loaded / e.total) * 40);
          setProgress(pct);
        },
      });

      setProgress(85);
      setStatus("Building results…");
      await new Promise((r) => setTimeout(r, 400));
      setProgress(100);

      onResult(res.data);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          err.message ||
          "Analysis failed. Is the backend running?",
      );
      setLoading(false);
      setProgress(0);
      setStatus("");
    }
  };

  return (
    <div className="page">
      <header className="header">
        <div className="logo">
          <div className="logo-dot" />
          DashGuard
        </div>
        <span className="header-tag">Accident Risk Analysis</span>
      </header>

      <main className="home-body">
        <div className="hero">
          <p className="hero-eyebrow">Deep Learning · Dashcam Analysis</p>
          <h1 className="hero-title">
            Predict risk
            <br />
            before <span>impact</span>
          </h1>
          <p className="hero-sub">
            Upload a dashcam video clip. Our spatiotemporal model scores each
            frame for accident risk and returns an annotated video with a full
            timeline.
          </p>
        </div>

        <div className="dropzone-wrap">
          <div
            className={`dropzone ${dragging ? "drag-over" : ""}`}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onClick={() => !loading && inputRef.current?.click()}
          >
            <input
              ref={inputRef}
              type="file"
              accept="video/*"
              onChange={(e) => handleFile(e.target.files?.[0])}
              disabled={loading}
            />
            <span className="dz-icon">🎬</span>
            <p className="dz-primary">Drop your dashcam footage here</p>
            <p className="dz-secondary">
              or click to browse &nbsp;·&nbsp; MP4, AVI, MOV supported
            </p>
          </div>

          {file && !loading && (
            <div className="file-chosen">
              <span>📹</span>
              <span className="file-chosen-name">{file.name}</span>
              <span className="file-chosen-size">{fmtBytes(file.size)}</span>
              <button className="file-clear" onClick={() => setFile(null)}>
                ×
              </button>
            </div>
          )}

          {error && <div className="error-box">⚠ {error}</div>}

          <div style={{ marginTop: 16 }}>
            <p style={{
              fontSize: 10, letterSpacing: 2, textTransform: "uppercase",
              opacity: 0.4, marginBottom: 10, fontFamily: "var(--font-mono)"
            }}>
              Select Model
            </p>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
              {[
                { id: "3dcnn",                  label: "3D CNN"},
                { id: "cnn_lstm",               label: "CNN + LSTM"},
                { id: "two_stream",             label: "Two-Stream"},
                { id: "cnn_transformer",        label: "CNN + Transformer"},
                { id: "two_stream_resnet",      label: "Two-Stream ResNet"},
                { id: "two_stream_transformer", label: "Two-Stream Transformer"},
              ].map((m) => (
                <button
                  key={m.id}
                  onClick={() => !loading && setSelectedModel(m.id)}
                  disabled={loading}
                  style={{
                    padding: "10px 8px",
                    fontFamily: "var(--font-mono)",
                    fontSize: 11,
                    border: `1.5px solid ${selectedModel === m.id ? "var(--ink)" : "var(--rule)"}`,
                    background: selectedModel === m.id ? "var(--ink)" : "transparent",
                    color: selectedModel === m.id ? "var(--paper)" : "var(--ink)",
                    borderRadius: "var(--radius)",
                    cursor: loading ? "not-allowed" : "pointer",
                    transition: "all 0.2s",
                    textAlign: "center",
                    lineHeight: 1.4,
                    opacity: loading ? 0.5 : 1,
                  }}
                >
                  <div style={{ fontWeight: 600 }}>{m.label}</div>
                  <div style={{ fontSize: 9, opacity: 0.55, marginTop: 2 }}>{m.desc}</div>
                </button>
              ))}
            </div>
          </div>
          <button
            className={`btn-analyze ${loading ? "loading" : ""}`}
            disabled={!file || loading}
            onClick={analyze}
          >
            {loading ? status || "Analyzing…" : "Analyze Video →"}
          </button>

          {loading && (
            <>
              <div className="progress-wrap">
                <div
                  className="progress-bar"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="progress-label">{progress}% complete</p>
            </>
          )}
        </div>
      </main>

      <footer className="footer">
        <span className="footer-note">
          6 models: 3D CNN · CNN+LSTM · Two-Stream · Transformer variants
        </span>
        <span className="footer-note">
          Local inference only — video never leaves your machine
        </span>
      </footer>
    </div>
  );
}

// ─── Result Page ──────────────────────────────────────────────────────────────
// Replace your entire ResultPage function with this

function ResultPage({ data, onBack }) {
  const { video, risk_scores, model_used, summary: backendSummary } = data;

  // Use backend summary if available, otherwise compute locally
  const summary = backendSummary ?? computeSummary(risk_scores);

  const fps = 10;
  const maxRisk   = summary?.peak_risk   ?? summary?.max    ?? 0;
  const avgRisk   = summary?.avg_risk    ?? summary?.avg    ?? 0;
  const alertFrms = summary?.alert_frames                   ?? 0;
  const totalFrms = summary?.total_frames ?? risk_scores?.length ?? 0;

  const alertTriggered = summary?.alert_triggered ?? maxRisk >= 0.7;
  const maxLevel = riskLevel(maxRisk);

  // First alert time
  const firstAlertIdx = risk_scores?.findIndex(s => s >= 0.7) ?? -1;
  const alertTime = firstAlertIdx >= 0
    ? `${(firstAlertIdx / fps).toFixed(1)}s`
    : "—";

  // Model display name
  const modelNames = {
    "3dcnn":                  "3D CNN",
    "cnn_lstm":               "CNN + LSTM",
    "two_stream":             "Two-Stream CNN",
    "cnn_transformer":        "CNN + Transformer",
    "two_stream_resnet":      "Two-Stream ResNet",
    "two_stream_transformer": "Two-Stream Transformer",
  };

  const analysisText = () => {
    if (alertTriggered) {
      return `The model detected a <strong>high-risk event</strong> at ${alertTime} into the clip, 
              with a peak risk score of <strong>${(maxRisk * 100).toFixed(0)}%</strong>. 
              A total of <strong>${alertFrms} frames</strong> exceeded the alert threshold (70%). 
              The average risk across the entire clip was ${(avgRisk * 100).toFixed(0)}%. 
              Review the annotated video above for the moment of highest risk.`;
    }
    return `No high-risk events were detected in this clip. 
            The peak risk score was <strong>${(maxRisk * 100).toFixed(0)}%</strong>, 
            below the 70% alert threshold. 
            The average risk score was ${(avgRisk * 100).toFixed(0)}% across ${totalFrms} evaluated frames. 
            The footage appears to show normal driving conditions.`;
  };

  return (
    <div className="page">
      <header className="header">
        <div className="logo">
          <div className="logo-dot" />
          DashGuard
        </div>
        <button className="btn-back" onClick={onBack}>← New Analysis</button>
      </header>

      <main className="result-body">
        {/* Title + badge */}
        <div style={{ display:"flex", alignItems:"center", gap:14, flexWrap:"wrap" }}>
          <h1 className="result-title">Analysis Results</h1>
          <span className={`result-badge ${alertTriggered ? "badge-alert" : "badge-ok"}`}>
            {alertTriggered ? "⚠ Alert Triggered" : "✓ No Alert"}
          </span>
        </div>

        {/* Summary stats */}
        <div className="summary-grid">
          <div className={`stat-card ${maxLevel.color}`}>
            <p className="stat-label">Peak Risk</p>
            <p className={`stat-value ${maxLevel.color}`}>
              {(maxRisk * 100).toFixed(0)}%
            </p>
          </div>
          <div className="stat-card neutral">
            <p className="stat-label">Avg Risk</p>
            <p className="stat-value">{(avgRisk * 100).toFixed(0)}%</p>
          </div>
          <div className={`stat-card ${alertTriggered ? "danger" : "safe"}`}>
            <p className="stat-label">Alert Frames</p>
            <p className={`stat-value ${alertTriggered ? "danger" : "safe"}`}>
              {alertFrms}
            </p>
          </div>
          <div className="stat-card neutral">
            <p className="stat-label">First Alert</p>
            <p className="stat-value" style={{ fontSize:22 }}>{alertTime}</p>
          </div>
          <div className="stat-card neutral">
            <p className="stat-label">Model</p>
            <p className="stat-value" style={{ fontSize:16, marginTop:6 }}>
              {modelNames[model_used] ?? model_used ?? "—"}
            </p>
          </div>
        </div>

        {/* Video + Chart */}
        <div className="content-grid">
          <div className="panel">
            <div className="panel-header"><span>▶</span> Processed Video</div>
            <div className="video-wrap">
              {video ? (
                <video
                  controls
                  src={`http://localhost:8000/video/${video}`}
                  onError={(e) => console.error("Video load error:", e)}
                />
              ) : (
                <div style={{
                  display:"flex", alignItems:"center", justifyContent:"center",
                  height:"100%", color:"rgba(255,255,255,0.3)",
                  fontSize:13, fontFamily:"var(--font-mono)"
                }}>
                  No video returned
                </div>
              )}
            </div>
          </div>

          <div className="panel chart-panel">
            <div className="panel-header"><span>◈</span> Risk Score Timeline</div>
            <div className="chart-wrap">
              {risk_scores?.length > 0
                ? <RiskChart scores={risk_scores} fps={fps} />
                : (
                  <p style={{ fontSize:12, opacity:0.4, padding:20 }}>
                    No score data available
                  </p>
                )
              }
            </div>
          </div>
        </div>

        {/* Analysis */}
        <div className="analysis-section">
          <h2 className="analysis-heading">Model Interpretation</h2>
          <p
            className="analysis-text"
            dangerouslySetInnerHTML={{ __html: analysisText() }}
          />
        </div>
      </main>

      <footer className="footer">
        <span className="footer-note">
          Threshold: 70% · FPS: {fps} · {totalFrms} frames evaluated
        </span>
        <span className="footer-note">
          Results are probabilistic — not a substitute for human judgment
        </span>
      </footer>
    </div>
  );
}

// ─── Root App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [result, setResult] = useState(null);

  // For demo/dev: inject mock data
  const mockAnalyze = () => {
    const n = 120;
    const scores = Array.from({ length: n }, (_, i) => {
      const t = i / n;
      const base = 0.08 + 0.12 * Math.sin(t * 4);
      const spike = i > 80 ? Math.min(1, ((i - 80) / 15) * 0.9) : 0;
      return Math.min(
        1,
        Math.max(0, base + spike + (Math.random() - 0.5) * 0.06),
      );
    });
    setResult({ video: null, risk_scores: scores });
  };

  return (
    <>
      <style>{css}</style>
      {result ? (
        <ResultPage data={result} onBack={() => setResult(null)} />
      ) : (
        <HomePage onResult={setResult} />
      )}
    </>
  );
}