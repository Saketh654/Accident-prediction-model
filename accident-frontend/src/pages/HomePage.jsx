/**
 * pages/HomePage.jsx
 * Upload page — drop zone, model selector, analyze button.
 * Props:
 *   onResult  (data) => void  — called with the API response on success
 */

import { useState } from "react";
import Header        from "../components/Header";
import Footer        from "../components/Footer";
import DropZone      from "../components/DropZone";
import ModelSelector from "../components/ModelSelector";
import { fmtBytes }  from "../utils/helpers";
import { useAnalyze } from "../hooks/useAnalyze";
import "../styles/home.css";

export default function HomePage({ onResult }) {
  const [file,          setFile]          = useState(null);
  const [fileError,     setFileError]     = useState("");
  const [selectedModel, setSelectedModel] = useState("3dcnn");

  const { loading, progress, status, error, analyze } = useAnalyze();

  // DropZone calls this with (file, errorMsg)
  const handleFile = (f, err) => {
    if (err) { setFileError(err); setFile(null); return; }
    setFile(f);
    setFileError("");
  };

  const handleAnalyze = () => {
    analyze(file, selectedModel, onResult);
  };

  const displayError = fileError || error;

  return (
    <div className="page">
      <Header />

      <main className="home-body">
        {/* Hero text */}
        <div className="hero">
          <p className="hero-eyebrow">Deep Learning · Dashcam Analysis</p>
          <h1 className="hero-title">
            Predict risk<br />before <span>impact</span>
          </h1>
          <p className="hero-sub">
            Upload a dashcam video clip. Our spatiotemporal model scores each
            frame for accident risk and returns an annotated video with a full
            timeline.
          </p>
        </div>

        {/* Upload area */}
        <div className="dropzone-wrap">
          <DropZone onFile={handleFile} disabled={loading} />

          {/* Selected file row */}
          {file && !loading && (
            <div className="file-chosen">
              <span>📹</span>
              <span className="file-chosen-name">{file.name}</span>
              <span className="file-chosen-size">{fmtBytes(file.size)}</span>
              <button className="file-clear" onClick={() => setFile(null)}>×</button>
            </div>
          )}

          {/* Error */}
          {displayError && (
            <div className="error-box">⚠ {displayError}</div>
          )}

          {/* Model picker */}
          <ModelSelector
            selected={selectedModel}
            onChange={setSelectedModel}
            disabled={loading}
          />

          {/* Analyze button */}
          <button
            className={`btn-analyze ${loading ? "loading" : ""}`}
            disabled={!file || loading}
            onClick={handleAnalyze}
          >
            {loading ? status || "Analyzing…" : "Analyze Video →"}
          </button>

          {/* Progress */}
          {loading && (
            <>
              <div className="progress-wrap">
                <div className="progress-bar" style={{ width: `${progress}%` }} />
              </div>
              <p className="progress-label">{progress}% complete</p>
            </>
          )}
        </div>
      </main>

      <Footer
        left="6 models: 3D CNN · CNN+LSTM · Two-Stream · Transformer variants"
        right="Local inference only — video never leaves your machine"
      />
    </div>
  );
}