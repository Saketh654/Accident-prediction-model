
import { MODEL_LIST } from "../utils/helpers";
import "../styles/home.css";

export default function ModelSelector({ selected, onChange, disabled }) {
  return (
    <div style={{ marginTop: 16 }}>
      <p className="model-select-label">Select Model</p>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
        {MODEL_LIST.map((m) => {
          const isActive = selected === m.id;
          return (
            <button
              key={m.id}
              disabled={disabled}
              onClick={() => !disabled && onChange(m.id)}
              className="model-btn"
              style={{
                border: `1.5px solid ${isActive ? "var(--ink)" : "var(--rule)"}`,
                background: isActive ? "var(--ink)" : "transparent",
                color:      isActive ? "var(--paper)" : "var(--ink)",
                cursor:     disabled ? "not-allowed" : "pointer",
                opacity:    disabled ? 0.5 : 1,
              }}
            >
              <div className="model-btn-label">{m.label}</div>
              <div className="model-btn-desc">{m.desc}</div>
            </button>
          );
        })}
      </div>
    </div>
  );
}