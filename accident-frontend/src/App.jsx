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