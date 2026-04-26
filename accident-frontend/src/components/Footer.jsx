/**
 * components/Footer.jsx
 * Shared bottom bar.
 * Props:
 *   left  string — left note text
 *   right string — right note text
 */

import "../styles/layout.css";

export default function Footer({ left, right }) {
  return (
    <footer className="footer">
      <span className="footer-note">{left}</span>
      <span className="footer-note">{right}</span>
    </footer>
  );
}