import { useState } from "react";
import axios from "axios";

const API_BASE = "http://localhost:8000";

export function useAnalyze() {
  const [loading,  setLoading]  = useState(false);
  const [progress, setProgress] = useState(0);
  const [status,   setStatus]   = useState("");
  const [error,    setError]    = useState("");

  async function analyze(file, modelName, onSuccess) {
    if (!file) return;

    setLoading(true);
    setProgress(0);
    setError("");
    setStatus("Uploading video…");

    const fd = new FormData();
    fd.append("file", file);
    fd.append("model_name", modelName);

    try {
      setProgress(15);
      setStatus("Running inference…");

      const res = await axios.post(`${API_BASE}/analyze`, fd, {
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

      onSuccess(res.data);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        err.message ||
        "Analysis failed. Is the backend running?"
      );
      setLoading(false);
      setProgress(0);
      setStatus("");
    }
  }

  return { loading, progress, status, error, analyze };
}