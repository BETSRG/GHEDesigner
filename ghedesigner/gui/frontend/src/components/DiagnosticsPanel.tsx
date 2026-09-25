import { AlertCircle, CheckCircle2, Wrench } from "lucide-react";

import type { Diagnostic } from "../types";

interface DiagnosticsPanelProps {
  diagnostics: Diagnostic[];
  validating: boolean;
  onNavigate?: (pointer: string) => void;
}

export function DiagnosticsPanel({ diagnostics, validating, onNavigate }: DiagnosticsPanelProps) {
  if (validating) {
    return <div className="diagnostic-summary pending">Checking the input…</div>;
  }
  if (diagnostics.length === 0) {
    return (
      <div className="diagnostic-summary valid">
        <CheckCircle2 size={18} />
        Input is valid
      </div>
    );
  }
  return (
    <div className="diagnostics-list">
      {diagnostics.map((diagnostic, index) => (
        <button
          className="diagnostic-card"
          key={`${diagnostic.pointer}-${diagnostic.message}-${index}`}
          onClick={() => onNavigate?.(diagnostic.pointer)}
          type="button"
        >
          <AlertCircle size={18} />
          <span>
            <strong>{diagnostic.pointer}</strong>
            <span>{diagnostic.message}</span>
            {diagnostic.suggestions[0] && (
              <small>
                <Wrench size={12} /> {diagnostic.suggestions[0]}
              </small>
            )}
          </span>
        </button>
      ))}
    </div>
  );
}
