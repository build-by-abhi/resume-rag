import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";

import App from "./App";
import "./index.css";

const container = document.getElementById("root");
if (!container) throw new Error("#root not found in index.html");

createRoot(container).render(
  <StrictMode>
    <BrowserRouter
      // Opt in to the v7 behaviours now. Without this the router logs a future
      // flag warning on every boot, which trains you to ignore the console -
      // and a noisy console is exactly how the 422 bug above stayed hidden.
      future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
    >
      <App />
    </BrowserRouter>
  </StrictMode>,
);