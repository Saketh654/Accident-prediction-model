
import "../styles/layout.css";

export default function Header({ onBack }) {
  return (
    <header className="header">
      <div className="logo">
        <div className="logo-dot" />
        DashGuard
      </div>

      {onBack ? (
        <button className="btn-back" onClick={onBack}>
          ← New Analysis
        </button>
      ) : (
        <span className="header-tag">Accident Risk Analysis</span>
      )}
    </header>
  );
}