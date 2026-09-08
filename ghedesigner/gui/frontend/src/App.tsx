import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  BookOpen,
  Braces,
  Building2,
  CheckCircle2,
  ChevronRight,
  CircleGauge,
  Database,
  Download,
  Droplets,
  FilePlus2,
  FolderOpen,
  GitFork,
  HardDrive,
  Leaf,
  PlayCircle,
  Redo2,
  Save,
  Settings2,
  ThermometerSun,
  Undo2,
  Waves,
  X,
} from "lucide-react";

import { api } from "./api";
import { DiagnosticsPanel } from "./components/DiagnosticsPanel";
import { CollectionSection } from "./components/CollectionSection";
import { JsonEditor } from "./components/JsonEditor";
import { NetworkEditor } from "./components/NetworkEditor";
import { Overview } from "./components/Overview";
import { SchemaSection } from "./components/SchemaSection";
import { RunSimulation } from "./components/RunSimulation";
import { WorkflowBar } from "./components/WorkflowBar";
import type { Diagnostic, ExampleSummary, InputDocument, JsonObject, WorkflowMode } from "./types";
import { blankDocument, deepClone, isJsonObject } from "./types";
import { useDocumentHistory } from "./useDocumentHistory";
import {
  applyWorkflow,
  collectionFields,
  inferWorkflow,
  seasonalGroundTemperatureRequired,
  simulationControlFields,
  soilFields,
  workflowDefinition,
  workflowDiagnostics,
  workflowSections,
} from "./workflows";

interface FileHandleLike {
  name: string;
  getFile(): Promise<File>;
  createWritable(): Promise<{ write(data: string): Promise<void>; close(): Promise<void> }>;
}

const sectionDefinitions = [
  { id: "overview", label: "Overview", icon: CircleGauge },
  { id: "fluid", label: "Fluid", icon: Droplets },
  { id: "soil", label: "Soil", icon: Leaf },
  { id: "simulation_control", label: "Simulation", icon: Settings2 },
  { id: "heat_pump", label: "HP Performance", icon: ThermometerSun },
  { id: "building", label: "Heat Pump Loads", icon: Building2 },
  { id: "ground_heat_exchanger", label: "GHEs", icon: Waves },
  { id: "horizontal_piping", label: "Horizontal Piping", icon: GitFork },
  { id: "source_sink_heat_exchanger", label: "Source / Sink", icon: Database },
  { id: "network", label: "Network", icon: GitFork },
  { id: "review", label: "Review & JSON", icon: Braces },
] as const;

const runDefinition = { id: "run", label: "Run Simulation", icon: PlayCircle } as const;

const sectionCopy: Record<string, { title: string; description: string }> = {
  fluid: {
    title: "Loop Fluid",
    description: "Set the circulation-fluid type, concentration, and representative design temperature.",
  },
  soil: {
    title: "Ground Properties",
    description:
      "Define shared thermal properties. The seasonal model appears only when horizontal-pipe simulation requires it.",
  },
  simulation_control: {
    title: "Simulation Controls",
    description: "Choose sizing or simulation duration, load treatment, design search, and horizontal-pipe behavior.",
  },
  heat_pump: {
    title: "Heat Pump Performance Library",
    description: "Maintain reusable performance maps referenced by installed heat pump/load components.",
  },
  building: {
    title: "Heat Pump and Load Components",
    description: "Define installed loads and heat pump behavior for each building connected to the district loop.",
  },
  ground_heat_exchanger: {
    title: "Ground Heat Exchangers",
    description: "Configure boreholes, pipes, grout, design flow per borehole, geometry, and local circulation pumps.",
  },
  horizontal_piping: {
    title: "Horizontal Piping Models",
    description: "Define buried-pipe models used by the ordered distribution segments in the district loop.",
  },
  source_sink_heat_exchanger: {
    title: "Source and Sink Heat Exchangers",
    description: "Maintain optional controlled source or sink exchangers for supported district workflows.",
  },
};

const collectionSections = new Set([
  "heat_pump",
  "building",
  "ground_heat_exchanger",
  "horizontal_piping",
  "source_sink_heat_exchanger",
]);

const parseDraft = (): InputDocument => {
  try {
    const stored = localStorage.getItem("ghedesigner.gui.draft");
    if (!stored) return blankDocument();
    const parsed: unknown = JSON.parse(stored);
    return isJsonObject(parsed) ? (parsed as InputDocument) : blankDocument();
  } catch {
    return blankDocument();
  }
};

const parseWorkflow = (): WorkflowMode | null => {
  const value = localStorage.getItem("ghedesigner.gui.workflow");
  return value === "standalone_design" ||
    value === "g_function" ||
    value === "building_design" ||
    value === "district_design" ||
    value === "district_simulation"
    ? value
    : null;
};

const downloadJson = (document: InputDocument, name: string) => {
  const blob = new Blob([`${JSON.stringify(document, null, 2)}\n`], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = window.document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
};

function App() {
  const [initialDocument] = useState(parseDraft);
  const history = useDocumentHistory(initialDocument);
  const [workflow, setWorkflow] = useState<WorkflowMode>(() => parseWorkflow() ?? inferWorkflow(initialDocument));
  const [schema, setSchema] = useState<JsonObject | null>(null);
  const [examples, setExamples] = useState<ExampleSummary[]>([]);
  const [diagnostics, setDiagnostics] = useState<Diagnostic[]>([]);
  const [validating, setValidating] = useState(true);
  const [section, setSection] = useState<string>("overview");
  const [examplesOpen, setExamplesOpen] = useState(false);
  const [openingExample, setOpeningExample] = useState<string | null>(null);
  const [fileName, setFileName] = useState("untitled-ghedesigner.json");
  const [fileHandle, setFileHandle] = useState<FileHandleLike | null>(null);
  const [savedSnapshot, setSavedSnapshot] = useState(() => JSON.stringify(history.document));
  const [status, setStatus] = useState("Recovered local draft");
  const [fatalError, setFatalError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const validationSequence = useRef(0);

  const documentSnapshot = useMemo(() => JSON.stringify(history.document), [history.document]);
  const dirty = documentSnapshot !== savedSnapshot;
  const visibleSectionIds = useMemo(
    () => workflowSections(history.document, workflow),
    [history.document, workflow],
  );
  const visibleSectionDefinitions = useMemo(
    () => sectionDefinitions.filter((item) => visibleSectionIds.includes(item.id)),
    [visibleSectionIds],
  );

  useEffect(() => {
    document.documentElement.setAttribute("data-ghedesigner-started", "true");
  }, []);

  useEffect(() => {
    localStorage.setItem("ghedesigner.gui.workflow", workflow);
  }, [workflow]);

  useEffect(() => {
    Promise.all([api.schema(), api.examples()])
      .then(([loadedSchema, loadedExamples]) => {
        setSchema(loadedSchema);
        setExamples(loadedExamples);
        api.clientLog("debug", "editor_initialized", { examples: loadedExamples.length });
      })
      .catch((caught: unknown) => {
        const message = caught instanceof Error ? caught.message : "Unable to load the GUI service.";
        api.clientLog("error", "editor_initialization_failed", { message });
        setFatalError(message);
      });
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem("ghedesigner.gui.draft", documentSnapshot);
    } catch (caught) {
      api.clientLog("error", "draft_autosave_failed", {
        message: caught instanceof Error ? caught.message : String(caught),
        bytes: documentSnapshot.length,
      });
    }
    const sequence = ++validationSequence.current;
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      setValidating(true);
      const started = performance.now();
      api
        .validate(history.document, controller.signal)
        .then((result) => {
          if (sequence !== validationSequence.current) return;
          const modeDiagnostics = workflowDiagnostics(history.document, workflow);
          setDiagnostics([...result.diagnostics, ...modeDiagnostics]);
          api.clientLog("debug", "validation_applied", {
            durationMs: performance.now() - started,
            diagnostics: result.diagnostics.length + modeDiagnostics.length,
            bytes: documentSnapshot.length,
          });
        })
        .catch((caught: unknown) => {
          if (controller.signal.aborted || sequence !== validationSequence.current) return;
          const message = caught instanceof Error ? caught.message : "Validation service failed.";
          api.clientLog("error", "validation_failed", { message, bytes: documentSnapshot.length });
          setDiagnostics([
            {
              severity: "error",
              message,
              pointer: "/",
              location: "Root",
              validator: "service",
              suggestions: [],
              source: "semantic",
            },
          ]);
        })
        .finally(() => {
          if (sequence === validationSequence.current) setValidating(false);
        });
    }, 300);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [documentSnapshot, history.document, workflow]);

  useEffect(() => {
    api.clientLog("debug", "section_opened", { section });
  }, [section]);

  useEffect(() => {
    if (section !== "run" && !visibleSectionIds.includes(section)) setSection("overview");
  }, [section, visibleSectionIds]);

  useEffect(() => {
    window.onbeforeunload = dirty ? () => true : null;
    return () => {
      window.onbeforeunload = null;
    };
  }, [dirty]);

  const loadDocument = useCallback(
    (document: InputDocument, name: string, handle: FileHandleLike | null = null) => {
      history.reset(document);
      setFileName(name);
      setFileHandle(handle);
      setSavedSnapshot(JSON.stringify(document));
      setStatus(`Opened ${name}`);
      setWorkflow(inferWorkflow(document));
      setSection("overview");
      api.clientLog("info", "document_loaded", {
        name,
        bytes: JSON.stringify(document).length,
        networkType: document.network?.type ?? "none",
      });
    },
    [history],
  );

  const readFile = useCallback(
    async (file: File, handle: FileHandleLike | null = null) => {
      try {
        const parsed: unknown = JSON.parse(await file.text());
        if (!isJsonObject(parsed)) throw new Error("The input root must be a JSON object.");
        loadDocument(parsed as InputDocument, file.name, handle);
      } catch (caught) {
        const message = caught instanceof Error ? caught.message : "Unable to open the JSON file.";
        api.clientLog("error", "document_open_failed", { name: file.name, message });
        window.alert(message);
      }
    },
    [loadDocument],
  );

  const openFile = async () => {
    const picker = (window as unknown as { showOpenFilePicker?: (options: unknown) => Promise<FileHandleLike[]> })
      .showOpenFilePicker;
    if (!picker) return fileInputRef.current?.click();
    try {
      const [handle] = await picker({
        types: [{ description: "GHEDesigner JSON", accept: { "application/json": [".json"] } }],
        multiple: false,
      });
      if (handle) await readFile(await handle.getFile(), handle);
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return;
      window.alert(caught instanceof Error ? caught.message : "Unable to open the file.");
    }
  };

  const save = useCallback(async () => {
    const text = `${JSON.stringify(history.document, null, 2)}\n`;
    try {
      let handle = fileHandle;
      if (!handle) {
        const picker = (window as unknown as { showSaveFilePicker?: (options: unknown) => Promise<FileHandleLike> })
          .showSaveFilePicker;
        if (picker) {
          handle = await picker({
            suggestedName: fileName,
            types: [{ description: "GHEDesigner JSON", accept: { "application/json": [".json"] } }],
          });
          setFileHandle(handle);
          setFileName(handle.name);
        } else {
          downloadJson(history.document, fileName);
        }
      }
      if (handle) {
        const writable = await handle.createWritable();
        await writable.write(text);
        await writable.close();
      }
      setSavedSnapshot(JSON.stringify(history.document));
      setStatus(`Saved ${handle?.name ?? fileName}`);
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return;
      const message = caught instanceof Error ? caught.message : "Unable to save the file.";
      api.clientLog("error", "document_save_failed", { name: fileName, message });
      window.alert(message);
    }
  }, [fileHandle, fileName, history.document]);

  useEffect(() => {
    const shortcut = (event: KeyboardEvent) => {
      if (!(event.metaKey || event.ctrlKey)) return;
      if (event.key.toLowerCase() === "s") {
        event.preventDefault();
        void save();
      } else if (event.key.toLowerCase() === "z" && !event.shiftKey) {
        event.preventDefault();
        history.undo();
      } else if (event.key.toLowerCase() === "y" || (event.key.toLowerCase() === "z" && event.shiftKey)) {
        event.preventDefault();
        history.redo();
      }
    };
    window.addEventListener("keydown", shortcut);
    return () => window.removeEventListener("keydown", shortcut);
  }, [history, save]);

  const newDocument = () => {
    if (dirty && !window.confirm("Discard the current unsaved draft?")) return;
    const document = applyWorkflow(blankDocument(), workflow);
    loadDocument(document, "untitled-ghedesigner.json");
    setStatus("Created a new input draft");
  };

  const changeWorkflow = (nextWorkflow: WorkflowMode) => {
    if (nextWorkflow === workflow) return;
    const leavingDistrict = workflow.startsWith("district_") && !nextWorkflow.startsWith("district_");
    const dropsBuildings =
      (nextWorkflow === "standalone_design" || nextWorkflow === "g_function") &&
      Object.keys(history.document.building ?? {}).length > 0;
    if (
      (leavingDistrict || dropsBuildings) &&
      !window.confirm(
        `Switch to ${workflowDefinition(nextWorkflow).label}? Components that are not used by that workflow will be removed from the input. You can undo this change.`,
      )
    ) return;
    history.update((document) => applyWorkflow(document, nextWorkflow));
    setWorkflow(nextWorkflow);
    setSection("overview");
    setStatus(`Switched to ${workflowDefinition(nextWorkflow).label}`);
    api.clientLog("info", "workflow_changed", { from: workflow, to: nextWorkflow });
  };

  const openExample = async (name: string) => {
    if (openingExample) return;
    setOpeningExample(name);
    setStatus(`Loading ${name}…`);
    const started = performance.now();
    try {
      const document = await api.example(name);
      loadDocument(document, name);
      setExamplesOpen(false);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Unable to load the example.";
      api.clientLog("error", "example_load_failed", { name, message, durationMs: performance.now() - started });
      setStatus(`Unable to load ${name}`);
      window.alert(message);
    } finally {
      setOpeningExample(null);
    }
  };

  const exportValidated = async () => {
    setValidating(true);
    try {
      const result = await api.validate(history.document);
      const modeDiagnostics = workflowDiagnostics(history.document, workflow);
      const allDiagnostics = [...result.diagnostics, ...modeDiagnostics];
      setDiagnostics(allDiagnostics);
      if (allDiagnostics.length > 0) {
        setSection("review");
        setStatus("Resolve validation errors before export");
        return;
      }
      downloadJson(history.document, fileName.replace(/\.json$/i, "") + ".validated.json");
      setStatus("Exported validated input JSON");
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Validation service failed during export.";
      api.clientLog("error", "validated_export_failed", { name: fileName, message });
      setStatus("Export failed; see the diagnostic log");
      window.alert(message);
    } finally {
      setValidating(false);
    }
  };

  const navigatePointer = (pointer: string) => {
    const root = pointer.split("/").filter(Boolean)[0];
    const known = visibleSectionDefinitions.some((item) => item.id === root);
    setSection(known ? root : "review");
  };

  const activeDefinition = visibleSectionDefinitions.find((item) => item.id === section);
  const content = useMemo(() => {
    if (!schema) return <div className="loading-screen">Loading the input schema…</div>;
    if (section === "run") {
      return (
        <RunSimulation
          document={history.document}
          diagnostics={diagnostics}
          validating={validating}
          fileName={fileName}
          onStatusChange={setStatus}
        />
      );
    }
    if (section === "overview") {
      return (
        <Overview
          document={history.document}
          workflow={workflow}
          onWorkflowChange={changeWorkflow}
          onOpenExamples={() => setExamplesOpen(true)}
          onNavigate={setSection}
        />
      );
    }
    if (section === "network") {
      return <NetworkEditor document={history.document} onChange={history.update} />;
    }
    if (section === "review") {
      return (
        <section className="editor-page review-page">
          <div className="page-heading">
            <div>
              <span className="eyebrow">Final Review</span>
              <h1>Validation and Source JSON</h1>
              <p>Resolve schema and topology diagnostics, or inspect and edit the complete document directly.</p>
            </div>
            <button type="button" className="button primary" disabled={validating || diagnostics.length > 0} onClick={() => void exportValidated()}>
              <Download size={16} /> Export valid input
            </button>
          </div>
          <div className="review-grid">
            <div className="form-surface diagnostics-surface">
              <div className="panel-heading compact-heading">
                <div>
                  <span className="eyebrow">Input Diagnostics</span>
                  <h2>{diagnostics.length ? `${diagnostics.length} issue${diagnostics.length === 1 ? "" : "s"}` : "Ready to export"}</h2>
                </div>
              </div>
              <DiagnosticsPanel diagnostics={diagnostics} validating={validating} onNavigate={navigatePointer} />
            </div>
            <JsonEditor value={history.document} onApply={history.update} />
          </div>
        </section>
      );
    }
    const copy = sectionCopy[section];
    if (copy) {
      if (collectionSections.has(section)) {
        return (
          <CollectionSection
            document={history.document}
            property={section}
            title={copy.title}
            description={
              section === "ground_heat_exchanger"
                ? workflow === "g_function" || workflow === "district_simulation"
                  ? "Enter each known borefield arrangement, borehole depth, construction, design flow, and component hydraulics."
                  : workflow === "standalone_design"
                    ? "Enter each GHE's construction, design flow, loads, geometric search constraints, and temperature limits."
                    : "Enter GHE construction, design flow, geometric search constraints, temperature limits, and component hydraulics."
                : copy.description
            }
            rootSchema={schema}
            allowedProperties={collectionFields(section, history.document, workflow)}
            workflow={workflow}
            onChange={history.update}
          />
        );
      }
      return (
        <SchemaSection
          document={deepClone(history.document)}
          rootSchema={schema}
          property={section}
          title={copy.title}
          description={
            section === "simulation_control"
              ? workflow === "district_design"
                ? "Set the system study duration, load treatment, GHE layout search, and optional horizontal-pipe fidelity."
                : workflow === "district_simulation"
                  ? "Set the system study duration and thermal fidelity for the supplied pre-designed borefields."
                  : "Set the number of years used to size the standalone GHE design."
              : copy.description
          }
          allowedProperties={
            section === "simulation_control"
              ? simulationControlFields(workflow)
              : section === "soil"
                ? soilFields(history.document)
                : undefined
          }
          requiredProperties={
            section === "soil" && seasonalGroundTemperatureRequired(history.document)
              ? ["ground_temperature_model"]
              : undefined
          }
          enumExclusions={
            section === "simulation_control" && workflow === "district_design"
              ? { search_method: ["SIMULATION_ONLY"] }
              : undefined
          }
          onChange={history.update}
        />
      );
    }
    return null;
  }, [schema, section, history.document, history.update, diagnostics, validating, fileName, workflow]);

  if (fatalError) {
    return (
      <main className="fatal-screen">
        <AlertTriangle size={36} />
        <h1>Unable to Start the Editor</h1>
        <p>{fatalError}</p>
      </main>
    );
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark"><Waves size={21} /></span>
          <span><strong>GHE</strong>Designer</span>
          <span className="product-label">Input Editor</span>
        </div>
        <div className="file-identity" title={fileName}>
          <span>{fileName}</span>
          {dirty && <i aria-label="unsaved changes" />}
        </div>
        <div className="topbar-actions">
          <button type="button" className="icon-button" title="Undo" disabled={!history.canUndo} onClick={history.undo}><Undo2 size={17} /></button>
          <button type="button" className="icon-button" title="Redo" disabled={!history.canRedo} onClick={history.redo}><Redo2 size={17} /></button>
          <span className="toolbar-divider" />
          <button type="button" className="button ghost compact" onClick={newDocument}><FilePlus2 size={16} /> New</button>
          <button type="button" className="button ghost compact" onClick={() => void openFile()}><FolderOpen size={16} /> Open</button>
          <button type="button" className="button secondary compact" onClick={() => void save()}><Save size={16} /> Save draft</button>
          <button type="button" className="button primary compact" onClick={() => void exportValidated()} disabled={validating || diagnostics.length > 0}>
            <Download size={16} /> Export
          </button>
        </div>
      </header>

      <WorkflowBar mode={workflow} onChange={changeWorkflow} />

      <aside className="sidebar">
        <nav>
          {visibleSectionDefinitions.map((item) => {
            const Icon = item.icon;
            const hasIssue = diagnostics.some((diagnostic) => diagnostic.pointer === "/" || diagnostic.pointer.startsWith(`/${item.id}`));
            return (
              <button
                type="button"
                key={item.id}
                className={section === item.id ? "active" : ""}
                onClick={() => setSection(item.id)}
              >
                <Icon size={17} />
                <span>{item.label}</span>
                {hasIssue && <i className="issue-dot" />}
                {section === item.id && <ChevronRight size={15} className="nav-chevron" />}
              </button>
            );
          })}
        </nav>
        <div className="sidebar-footer">
          <button
            type="button"
            className={section === runDefinition.id ? "active run-nav-button" : "run-nav-button"}
            onClick={() => setSection(runDefinition.id)}
          >
            <PlayCircle size={17} /> <span>Run simulation</span>
            {diagnostics.length > 0 && <i className="issue-dot" />}
            {section === runDefinition.id && <ChevronRight size={15} className="nav-chevron" />}
          </button>
          <button type="button" onClick={() => setExamplesOpen(true)}><BookOpen size={16} /> Examples</button>
          <div className={diagnostics.length ? "validation-badge invalid" : "validation-badge"}>
            {diagnostics.length ? <AlertTriangle size={15} /> : <CheckCircle2 size={15} />}
            {validating ? "Validating…" : diagnostics.length ? `${diagnostics.length} issues` : "Valid input"}
          </div>
        </div>
      </aside>

      <main className="main-content" aria-label={section === "run" ? runDefinition.label : activeDefinition?.label}>{content}</main>
      <footer className="statusbar"><HardDrive size={13} /> {status}</footer>

      <input
        ref={fileInputRef}
        type="file"
        accept="application/json,.json"
        hidden
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) void readFile(file);
          event.target.value = "";
        }}
      />

      {examplesOpen && (
        <div className="modal-backdrop" role="presentation" onMouseDown={() => setExamplesOpen(false)}>
          <section className="modal" role="dialog" aria-modal="true" aria-label="Example input files" onMouseDown={(event) => event.stopPropagation()}>
            <div className="panel-heading">
              <div><span className="eyebrow">Maintained Inputs</span><h2>Start From an Example</h2></div>
              <button type="button" className="icon-button" onClick={() => setExamplesOpen(false)}><X size={18} /></button>
            </div>
            <div className="example-list">
              {examples.filter((example) => example.workflow_mode === workflow).map((example) => (
                <button
                  type="button"
                  key={example.name}
                  disabled={openingExample !== null}
                  onClick={() => void openExample(example.name)}
                >
                  <span><strong>{example.label}</strong><small>{example.name}</small></span>
                  <span className="example-type">
                    {openingExample === example.name ? "loading…" : example.network_type?.replaceAll("_", " ") ?? "standalone"}
                  </span>
                </button>
              ))}
              {!examples.some((example) => example.workflow_mode === workflow) && (
                <p className="empty-inline">No maintained examples currently match this workflow.</p>
              )}
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

export default App;
