simport Header    from "../components/Header";
import Footer    from "../components/Footer";
import RiskChart from "../components/RiskChart";
import { riskLevel, computeSummary, MODEL_NAMES } from "../utils/helpers";
import "../styles/result.css";

const API_BASE = "http://localhost:8000";
const FPS      = 10;

export default function ResultPage({ data, onBack }) {
  const { video, risk_scores, model_used, summary: backendSummary } = data;

  // Prefer backend summary, fall back to local computation
  const summary       = backendSummary ?? computeSummary(risk_scores);
  const maxRisk       = summary?.peak_risk    ?? summary?.max ?? 0;
  const avgRisk       = summary?.avg_risk     ?? summary?.avg ?? 0;
  const alertFrms     = summary?.alert_frames ?? 0;
  const totalFrms     = summary?.total_frames ?? risk_scores?.length ?? 0;
  const alertTriggered = summary?.alert_triggered ?? maxRisk >= 0.7;
  const maxLevel      = riskLevel(maxRisk);

  const firstAlertIdx = risk_scores?.findIndex((s) => s >= 0.7) ?? -1;
  const alertTime     = firstAlertIdx >= 0
    ? `${(firstAlertIdx / FPS).toFixed(1)}s`
    : "—";

  const analysisText = alertTriggered
    ? `The model detected a <strong>high-risk event</strong> at ${alertTime} into the clip,
       with a peak risk score of <strong>${(maxRisk * 100).toFixed(0)}%</strong>.
       A total of <strong>${alertFrms} frames</strong> exceeded the alert threshold (70%).
       The average risk across the entire clip was ${(avgRisk * 100).toFixed(0)}%.
       Review the annotated video above for the moment of highest risk.`
    : `No high-risk events were detected in this clip.
       The peak risk score was <strong>${(maxRisk * 100).toFixed(0)}%</strong>,
       below the 70% alert threshold.
       The average risk score was ${(avgRisk * 100).toFixed(0)}% across ${totalFrms} evaluated frames.
       The footage appears to show normal driving conditions.`;

  return (
    <div className="page">
      <Header onBack={onBack} />

      <main className="result-body">

        {/* Title + alert badge */}
        <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
          <h1 className="result-title">Analysis Results</h1>
          <span className={`result-badge ${alertTriggered ? "badge-alert" : "badge-ok"}`}>
            {alertTriggered ? "⚠ Alert Triggered" : "✓ No Alert"}
          </span>
        </div>

        {/* Stat cards */}
        <div className="summary-grid">
          <StatCard label="Peak Risk"    accent={maxLevel.color}>
            <span className={`stat-value ${maxLevel.color}`}>
              {(maxRisk * 100).toFixed(0)}%
            </span>
          </StatCard>

          <StatCard label="Avg Risk" accent="neutral">
            <span className="stat-value">{(avgRisk * 100).toFixed(0)}%</span>
          </StatCard>

          <StatCard label="Alert Frames" accent={alertTriggered ? "danger" : "safe"}>
            <span className={`stat-value ${alertTriggered ? "danger" : "safe"}`}>
              {alertFrms}
            </span>
          </StatCard>

          <StatCard label="First Alert" accent="neutral">
            <span className="stat-value" style={{ fontSize: 22 }}>{alertTime}</span>
          </StatCard>

          <StatCard label="Model" accent="neutral">
            <span className="stat-value" style={{ fontSize: 15, marginTop: 6 }}>
              {MODEL_NAMES[model_used] ?? model_used ?? "—"}
            </span>
          </StatCard>
        </div>

        {/* Video + Chart */}
        <div className="content-grid">
          {/* Video panel */}
          <div className="panel">
            <div className="panel-header"><span>▶</span> Processed Video</div>
            <div className="video-wrap">
              {video ? (
                <video
                  controls
                  src={`${API_BASE}/video/${video}`}
                  onError={(e) => console.error("Video load error:", e)}
                />
              ) : (
                <div style={{
                  display: "flex", alignItems: "center", justifyContent: "center",
                  height: "100%", color: "rgba(255,255,255,0.3)",
                  fontSize: 13, fontFamily: "var(--font-mono)",
                }}>
                  No video returned
                </div>
              )}
            </div>
          </div>

          {/* Chart panel */}
          <div className="panel chart-panel">
            <div className="panel-header"><span>◈</span> Risk Score Timeline</div>
            <div className="chart-wrap">
              {risk_scores?.length > 0
                ? <RiskChart scores={risk_scores} fps={FPS} />
                : <p style={{ fontSize: 12, opacity: 0.4, padding: 20 }}>No score data available</p>
              }
            </div>
          </div>
        </div>

        {/* Analysis text */}
        <div className="analysis-section">
          <h2 className="analysis-heading">Model Interpretation</h2>
          <p
            className="analysis-text"
            dangerouslySetInnerHTML={{ __html: analysisText }}
          />
        </div>

      </main>

      <Footer
        left={`Threshold: 70% · FPS: ${FPS} · ${totalFrms} frames evaluated`}
        right="Results are probabilistic — not a substitute for human judgment"
      />
    </div>
  );
}

// ── Small inline helper ────────────────────────────────────────────────────────
function StatCard({ label, accent, children }) {
  return (
    <div className={`stat-card ${accent}`}>
      <p className="stat-label">{label}</p>
      {children}
    </div>
  );
}