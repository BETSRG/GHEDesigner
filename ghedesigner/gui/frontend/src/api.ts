import type {
  ExampleSummary,
  InputDocument,
  JsonObject,
  NetworkPreview,
  PathSelection,
  RunSettings,
  SimulationJob,
  ValidationResult,
} from "./types";

type ClientLogLevel = "debug" | "info" | "warning" | "error";

const clientLog = (level: ClientLogLevel, event: string, details: Record<string, unknown> = {}) => {
  const body = JSON.stringify({ level, event, details });
  try {
    if (navigator.sendBeacon?.("/api/client-log", new Blob([body], { type: "application/json" }))) return;
    void fetch("/api/client-log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      keepalive: true,
    });
  } catch {
    // Logging must never interfere with the editor.
  }
};

const jsonRequest = async <T,>(url: string, init: RequestInit = {}, timeoutMs = 30_000): Promise<T> => {
  const controller = new AbortController();
  const externalSignal = init.signal;
  let timedOut = false;
  const abortFromCaller = () => controller.abort();
  if (externalSignal?.aborted) controller.abort();
  else externalSignal?.addEventListener("abort", abortFromCaller, { once: true });
  const timeout = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const started = performance.now();

  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    const payload = (await response.json()) as T & { error?: string };
    if (!response.ok) {
      throw new Error(payload.error ?? `Request failed with status ${response.status}.`);
    }
    const durationMs = performance.now() - started;
    if (durationMs >= 2_000) {
      clientLog("warning", "slow_api_request", { url, method: init.method ?? "GET", durationMs });
    }
    return payload;
  } catch (caught) {
    if (timedOut) {
      clientLog("error", "api_request_timeout", { url, method: init.method ?? "GET", timeoutMs });
      throw new Error(`The GUI service did not respond within ${Math.round(timeoutMs / 1_000)} seconds.`);
    }
    if (!externalSignal?.aborted) {
      clientLog("error", "api_request_failed", {
        url,
        method: init.method ?? "GET",
        message: caught instanceof Error ? caught.message : String(caught),
      });
    }
    throw caught;
  } finally {
    window.clearTimeout(timeout);
    externalSignal?.removeEventListener("abort", abortFromCaller);
  }
};

export const api = {
  clientLog,
  schema: () => jsonRequest<JsonObject>("/api/schema"),
  examples: () => jsonRequest<ExampleSummary[]>("/api/examples"),
  example: (name: string) => jsonRequest<InputDocument>(`/api/examples/${encodeURIComponent(name)}`),
  runSettings: () => jsonRequest<RunSettings>("/api/run-settings"),
  choosePath: (kind: "file" | "directory", initialPath = "") =>
    jsonRequest<PathSelection>(
      "/api/choose-path",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind, initial_path: initialPath }),
      },
      600_000,
    ),
  openDirectory: (path: string) =>
    jsonRequest<PathSelection>("/api/open-directory", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    }),
  validate: (document: InputDocument, signal?: AbortSignal) =>
    jsonRequest<ValidationResult>("/api/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(document),
      signal,
    }),
  compileNetwork: (document: InputDocument, signal?: AbortSignal) =>
    jsonRequest<NetworkPreview>("/api/compile-network", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(document),
      signal,
    }),
  startSimulation: (document: InputDocument, outputDirectory: string, inputName: string) =>
    jsonRequest<SimulationJob>("/api/simulations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document, output_directory: outputDirectory, input_name: inputName }),
    }),
  simulationStatus: (jobId: string) =>
    jsonRequest<SimulationJob>(`/api/simulations/${encodeURIComponent(jobId)}`),
  cancelSimulation: (jobId: string) =>
    jsonRequest<SimulationJob>(`/api/simulations/${encodeURIComponent(jobId)}`, { method: "DELETE" }),
};
