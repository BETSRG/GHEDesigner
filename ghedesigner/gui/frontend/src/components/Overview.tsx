import { ArrowRight, Calculator, DraftingCompass, Layers3, Network, PlayCircle, Waypoints } from "lucide-react";

import type { InputDocument, WorkflowMode } from "../types";
import { workflowDefinition, workflowDefinitions } from "../workflows";

interface OverviewProps {
  document: InputDocument;
  workflow: WorkflowMode;
  onWorkflowChange: (workflow: WorkflowMode) => void;
  onOpenExamples: () => void;
  onNavigate: (section: string) => void;
}

const workflowIcons = {
  standalone_design: DraftingCompass,
  g_function: Calculator,
  building_design: Waypoints,
  district_design: Network,
  district_simulation: PlayCircle,
};

export function Overview({ document, workflow, onWorkflowChange, onOpenExamples, onNavigate }: OverviewProps) {
  const selected = workflowDefinition(workflow);
  const networkType = typeof document.network?.type === "string" ? document.network.type : "not configured";
  const counts = {
    buildings: Object.keys(document.building ?? {}).length,
    ghes: Object.keys(document.ground_heat_exchanger ?? {}).length,
  };

  return (
    <section className="editor-page overview-page">
      <div className="hero-card">
        <div>
          <span className="eyebrow">Selected Workflow · Version {String(document.version ?? 4)}</span>
          <h1>{selected.label}</h1>
          <p>{selected.description}</p>
          <div className="button-row">
            <button type="button" className="button primary" onClick={() => onNavigate("fluid")}>
              Begin this workflow <ArrowRight size={16} />
            </button>
            <button type="button" className="button secondary" onClick={onOpenExamples}>
              <Layers3 size={16} /> Relevant examples
            </button>
          </div>
        </div>
        <div className="hero-mark" aria-hidden="true">
          {(() => {
            const Icon = workflowIcons[workflow];
            return <Icon size={58} />;
          })()}
        </div>
      </div>

      <div className="workflow-picker-heading">
        <div>
          <span className="eyebrow">Start Page</span>
          <h2>Choose What This Input File Should Do</h2>
        </div>
        <p>The choice controls the navigation and exposes only fields used by that execution path.</p>
      </div>

      <div className="workflow-choice-grid">
        {workflowDefinitions.map((definition) => {
          const Icon = workflowIcons[definition.id];
          return (
            <button
              type="button"
              key={definition.id}
              className={workflow === definition.id ? "workflow-choice active" : "workflow-choice"}
              onClick={() => onWorkflowChange(definition.id)}
            >
              <span className="workflow-choice-icon"><Icon size={21} /></span>
              <strong>{definition.label}</strong>
              <span>{definition.description}</span>
              <small>{workflow === definition.id ? "Selected workflow" : "Select workflow"}</small>
            </button>
          );
        })}
      </div>

      <div className="workflow-detail-grid">
        <article className="form-surface workflow-sequence">
          <span className="eyebrow">Workflow Sequence</span>
          <h2>What Happens Next</h2>
          <ol>
            {selected.steps.map((step) => <li key={step}>{step}</li>)}
          </ol>
        </article>
        <article className="form-surface workflow-outcome">
          <span className="eyebrow">Result</span>
          <h2>What This Produces</h2>
          <p>{selected.outcome}</p>
          <dl>
            <div><dt>GHEs</dt><dd>{counts.ghes}</dd></div>
            {(workflow === "building_design" || workflow.startsWith("district_")) && (
              <div><dt>Buildings</dt><dd>{counts.buildings}</dd></div>
            )}
            {workflow.startsWith("district_") && (
              <div><dt>Topology</dt><dd>{networkType.replaceAll("_", " ")}</dd></div>
            )}
          </dl>
        </article>
      </div>
    </section>
  );
}
