import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { api } from "./api";
import "./styles.css";

const errorDetails = (value: unknown) => {
  if (value instanceof Error) return { name: value.name, message: value.message, stack: value.stack };
  return { message: String(value) };
};

window.addEventListener("error", (event) => {
  api.clientLog("error", "browser_error", {
    ...errorDetails(event.error ?? event.message),
    filename: event.filename,
    line: event.lineno,
    column: event.colno,
  });
});

window.addEventListener("unhandledrejection", (event) => {
  api.clientLog("error", "unhandled_promise_rejection", errorDetails(event.reason));
});

api.clientLog("info", "browser_started", {
  userAgent: navigator.userAgent,
  language: navigator.language,
});

let lastHeartbeat = performance.now();
window.setInterval(() => {
  const now = performance.now();
  const delayMs = now - lastHeartbeat - 1_000;
  lastHeartbeat = now;
  if (document.visibilityState === "visible" && delayMs >= 2_000) {
    api.clientLog("warning", "browser_event_loop_stall", { delayMs });
  }
}, 1_000);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
