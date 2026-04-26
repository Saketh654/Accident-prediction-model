/**
 * utils/helpers.js
 * Pure utility functions — no React, no side effects.
 */

export function fmtBytes(b) {
  if (b < 1024)            return `${b} B`;
  if (b < 1024 * 1024)     return `${(b / 1024).toFixed(1)} KB`;
  return                          `${(b / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Returns a risk level descriptor based on the score value.
 * @param {number} v  0–1 risk score
 */
export function riskLevel(v) {
  if (v >= 0.7) return { label: "HIGH",   color: "danger", accent: "#e63323" };
  if (v >= 0.4) return { label: "MEDIUM", color: "warn",   accent: "#e8a020" };
  return               { label: "LOW",    color: "safe",   accent: "#2d9e6b" };
}

/**
 * Compute a summary object from a raw scores array.
 * Used as fallback when the backend doesn't return a summary field.
 * @param {number[]} scores
 */
export function computeSummary(scores) {
  if (!scores || scores.length === 0) return null;
  const max         = Math.max(...scores);
  const avg         = scores.reduce((a, b) => a + b, 0) / scores.length;
  const alertFrames = scores.filter((s) => s >= 0.7).length;
  const firstAlert  = scores.findIndex((s) => s >= 0.7);
  return { max, avg, alertFrames, firstAlert, total: scores.length };
}

/**
 * Human-readable model display names.
 */
export const MODEL_NAMES = {
  "3dcnn":                  "3D CNN",
  "cnn_lstm":               "CNN + LSTM",
  "two_stream":             "Two-Stream CNN",
  "cnn_transformer":        "CNN + Transformer",
  "two_stream_resnet":      "Two-Stream ResNet",
  "two_stream_transformer": "Two-Stream Transformer",
};

/**
 * Model selector list — id, label, and short description.
 */
export const MODEL_LIST = [
  { id: "3dcnn",                  label: "3D CNN",            desc: "Fastest"    },
  { id: "cnn_lstm",               label: "CNN + LSTM",        desc: "Balanced"   },
  { id: "two_stream",             label: "Two-Stream",        desc: "Flow-aware" },
  { id: "cnn_transformer",        label: "CNN + Transformer", desc: "Attention"  },
  { id: "two_stream_resnet",      label: "Two-Stream ResNet", desc: "Pretrained" },
  { id: "two_stream_transformer", label: "Two-Stream Trans.", desc: "Best"       },
];