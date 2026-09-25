import {
  AlertTriangle,
  CheckCircle2,
  Clock3,
  FolderOpen,
  FolderOutput,
  PlayCircle,
  Square,
  XCircle,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { api } from "../api";
import type { Diagnostic, InputDocument, SimulationJob } from "../types";
import { PathBrowserButton } from "./PathBrowser";

const ACTIVE_STATUSES = new Set(["queued", "running", "cancelling"]);
const JOB_STORAGE_KEY = "ghedesigner.gui.simulation-job";
const OUTPUT_STORAGE_KEY = "ghedesigner.gui.output-directory";

interface RunSimulationProps {
  document: InputDocument;
  diagnostics: Diagnostic[];
  validating: boolean;
  fileName: string;
  onStatusChange: (status: string) => void;
}

const statusIcon = (job: SimulationJob) => {
  if (job.status === "completed") return <CheckCircle2 size={18} />;
  if (job.status === "failed" || job.status === "cancelled") return <XCircle size={18} />;
  return <Clock3 size={18} />;
};

const statusLabel = (status: SimulationJob["status"]) =>
  ({
    queued: "Queued",
    running: "Running",
    cancelling: "Cancelling",
    completed: "Completed",
    failed: "Failed",
    cancelled: "Cancelled",
  })[status];

export function RunSimulation({
  document,
  diagnostics,
  validating,
  fileName,
  onStatusChange,
}: RunSimulationProps) {
  const [outputDirectory, setOutputDirectory] = useState(() => localStorage.getItem(OUTPUT_STORAGE_KEY) ?? "");
  const [job, setJob] = useState<SimulationJob | null>(null);
  const [starting, setStarting] = useState(false);
  const [openingDirectory, setOpeningDirectory] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const active = job !== null && ACTIVE_STATUSES.has(job.status);
  const canRun = !validating && diagnostics.length === 0 && outputDirectory.trim().length > 0 && !active && !starting;

  useEffect(() => {
    const storedJobId = localStorage.getItem(JOB_STORAGE_KEY);
    const settings = outputDirectory ? Promise.resolve(null) : api.runSettings();
    const previousJob = storedJobId ? api.simulationStatus(storedJobId) : Promise.resolve(null);
    Promise.allSettled([settings, previousJob]).then(([settingsResult, jobResult]) => {
      if (settingsResult.status === "fulfilled" && settingsResult.value) {
        setOutputDirectory(settingsResult.value.default_output_directory);
      }
      if (jobResult.status === "fulfilled" && jobResult.value) {
        setJob(jobResult.value);
      } else if (storedJobId) {
        localStorage.removeItem(JOB_STORAGE_KEY);
      }
    });
  }, []);

  useEffect(() => {
    localStorage.setItem(OUTPUT_STORAGE_KEY, outputDirectory);
  }, [outputDirectory]);

  useEffect(() => {
    if (!job || !ACTIVE_STATUSES.has(job.status)) return;
    let disposed = false;
    let timer: number | undefined;
    const poll = async () => {
      try {
        const current = await api.simulationStatus(job.id);
        if (disposed) return;
        setError(null);
        setJob(current);
        if (ACTIVE_STATUSES.has(current.status)) {
          timer = window.setTimeout(() => void poll(), 750);
        } else {
          onStatusChange(
            current.status === "completed" ? "Simulation completed" : `Simulation ${current.status}`,
          );
        }
      } catch (caught) {
        if (disposed) return;
        setError(caught instanceof Error ? caught.message : "Unable to read simulation status.");
        timer = window.setTimeout(() => void poll(), 2_000);
      }
    };
    timer = window.setTimeout(() => void poll(), 400);
    return () => {
      disposed = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [job?.id, job?.status, onStatusChange]);

  const duration = useMemo(() => {
    if (!job?.started_at) return null;
    const end = job.finished_at ? Date.parse(job.finished_at) : Date.now();
    const seconds = Math.max(0, Math.round((end - Date.parse(job.started_at)) / 1_000));
    return seconds < 60 ? `${seconds} s` : `${Math.floor(seconds / 60)} min ${seconds % 60} s`;
  }, [job]);

  const start = async () => {
    setStarting(true);
    setError(null);
    onStatusChange("Starting simulation…");
    try {
      const started = await api.startSimulation(document, outputDirectory.trim(), fileName);
      setJob(started);
      localStorage.setItem(JOB_STORAGE_KEY, started.id);
      onStatusChange("Simulation running");
      api.clientLog("info", "simulation_started_from_editor", {
        jobId: started.id,
        outputDirectory: started.output_directory,
      });
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Unable to start the simulation.";
      setError(message);
      onStatusChange("Simulation could not start");
    } finally {
      setStarting(false);
    }
  };

  const cancel = async () => {
    if (!job || !ACTIVE_STATUSES.has(job.status)) return;
    setError(null);
    try {
      const cancelled = await api.cancelSimulation(job.id);
      setJob(cancelled);
      onStatusChange("Cancelling simulation…");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to cancel the simulation.");
    }
  };

  const openOutputDirectory = async () => {
    const path = outputDirectory.trim();
    if (!path) return;
    setOpeningDirectory(true);
    setError(null);
    try {
      const opened = await api.openDirectory(path);
      if (opened.path) setOutputDirectory(opened.path);
      onStatusChange("Opened output directory");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to open the output directory.");
      onStatusChange("Output directory could not be opened");
    } finally {
      setOpeningDirectory(false);
    }
  };

  return (
    <section className="editor-page run-page">
      <div className="page-heading">
        <div>
          <span className="eyebrow">Execution</span>
          <h1>Run the Simulation</h1>
          <p>Choose an output directory on the computer running this GUI, then execute the current validated input.</p>
        </div>
      </div>

      <div className="run-grid">
        <section className="form-surface run-configuration">
          <div className="panel-heading compact-heading">
            <div>
              <span className="eyebrow">Destination</span>
              <h2>Output Directory</h2>
            </div>
            <FolderOutput size={24} />
          </div>
          <div className="run-directory-field">
            <label htmlFor="simulation-output-directory">Directory Path</label>
            <div className="path-picker">
              <input
                id="simulation-output-directory"
                type="text"
                value={outputDirectory}
                disabled={active || starting}
                spellCheck={false}
                placeholder="/path/to/simulation-output"
                onChange={(event) => setOutputDirectory(event.target.value)}
              />
              <PathBrowserButton
                kind="directory"
                currentPath={outputDirectory}
                disabled={active || starting}
                onSelect={setOutputDirectory}
              />
            </div>
            <small>Relative paths are resolved from the directory where `ghedesigner-gui` was launched.</small>
          </div>

          {diagnostics.length > 0 && (
            <div className="run-readiness invalid">
              <AlertTriangle size={17} />
              Resolve {diagnostics.length} input issue{diagnostics.length === 1 ? "" : "s"} before running.
            </div>
          )}
          {!diagnostics.length && !validating && (
            <div className="run-readiness valid">
              <CheckCircle2 size={17} /> The current input passes schema and topology validation.
            </div>
          )}
          {validating && <div className="run-readiness pending"><Clock3 size={17} /> Checking the current input…</div>}
          {error && <div className="run-readiness invalid"><AlertTriangle size={17} /> {error}</div>}

          <div className="run-actions">
            <button type="button" className="button primary" disabled={!canRun} onClick={() => void start()}>
              <PlayCircle size={17} /> {starting ? "Starting…" : "Run simulation"}
            </button>
            <button
              type="button"
              className="button secondary"
              disabled={!outputDirectory.trim() || openingDirectory}
              onClick={() => void openOutputDirectory()}
            >
              <FolderOpen size={17} /> {openingDirectory ? "Opening…" : "Open Output Directory"}
            </button>
            {active && (
              <button type="button" className="button danger" onClick={() => void cancel()}>
                <Square size={15} /> Cancel
              </button>
            )}
          </div>
        </section>

        <section className="form-surface run-status-panel">
          <div className="panel-heading compact-heading">
            <div>
              <span className="eyebrow">Process</span>
              <h2>Simulation Status</h2>
            </div>
            {job && <span className={`run-status ${job.status}`}>{statusIcon(job)} {statusLabel(job.status)}</span>}
          </div>
          {job ? (
            <>
              <dl className="run-metadata">
                <div><dt>Input</dt><dd>{job.input_name}</dd></div>
                <div><dt>Output</dt><dd title={job.output_directory}>{job.output_directory}</dd></div>
                {duration && <div><dt>Elapsed</dt><dd>{duration}</dd></div>}
                {job.return_code !== null && <div><dt>Exit status</dt><dd>{job.return_code}</dd></div>}
              </dl>
              <pre className="process-output" aria-live="polite">{job.output || "Waiting for process output…"}</pre>
            </>
          ) : (
            <div className="run-empty-state">
              <PlayCircle size={34} />
              <p>No simulation has been started in this GUI session.</p>
            </div>
          )}
        </section>
      </div>
    </section>
  );
}
