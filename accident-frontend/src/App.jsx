/**
 * App.jsx — root component
 * Only responsible for routing between HomePage and ResultPage.
 * All logic lives in pages/, components/, hooks/, and utils/.
 */

import { useState } from "react";
import HomePage   from "./pages/HomePage";
import ResultPage from "./pages/ResultPage";
import "./styles/global.css";

export default function App() {
  const [result, setResult] = useState(null);

  return result
    ? <ResultPage data={result} onBack={() => setResult(null)} />
    : <HomePage   onResult={setResult} />;
}