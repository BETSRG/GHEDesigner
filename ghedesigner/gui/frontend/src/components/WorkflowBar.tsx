import { Calculator, DraftingCompass, Network, PlayCircle, Waypoints } from "lucide-react";

import type { WorkflowMode } from "../types";
import { workflowDefinitions } from "../workflows";

const icons = {
  standalone_design: DraftingCompass,
  g_function: Calculator,
  building_design: Waypoints,
  district_design: Network,
  district_simulation: PlayCircle,
};

interface WorkflowBarProps {
  mode: WorkflowMode;
  onChange: (mode: WorkflowMode) => void;
}

export function WorkflowBar({ mode, onChange }: WorkflowBarProps) {
  return (
    <div className="workflow-bar" aria-label="Simulation workflow">
      <span className="workflow-bar-label">Workflow</span>
      <div className="workflow-tabs" role="tablist" aria-label="GHEDesigner workflow">
        {workflowDefinitions.map((definition) => {
          const Icon = icons[definition.id];
          return (
            <button
              type="button"
              role="tab"
              aria-selected={mode === definition.id}
              className={mode === definition.id ? "active" : ""}
              key={definition.id}
              title={definition.description}
              onClick={() => onChange(definition.id)}
            >
              <Icon size={15} />
              <span>{definition.shortLabel}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
