import { Network } from "lucide-react";

import { createCompactNetwork } from "../networkUtils";
import type { InputDocument } from "../types";
import { CompactNetworkEditor } from "./CompactNetworkEditor";

interface NetworkEditorProps {
  document: InputDocument;
  onChange: (document: InputDocument) => void;
}

export function NetworkEditor({ document, onChange }: NetworkEditorProps) {
  const type = document.network?.type;

  if (type !== "one_pipe" && type !== "two_pipe") {
    return (
      <section className="editor-page">
        <div className="page-heading">
          <div>
            <span className="eyebrow">Topology</span>
            <h1>Choose a Network Topology</h1>
            <p>Choose an ordered, unidirectional one-pipe or two-pipe distribution system.</p>
          </div>
        </div>
        <div className="network-choice-grid">
          <button type="button" onClick={() => onChange({ ...document, network: createCompactNetwork(document, "one_pipe") })}>
            <Network size={30} />
            <strong>One-Pipe Loop</strong>
            <span>Ordered stations, station bypasses, and controlled distribution circulation.</span>
          </button>
          <button type="button" onClick={() => onChange({ ...document, network: createCompactNetwork(document, "two_pipe") })}>
            <Network size={30} />
            <strong>Two-Pipe Network</strong>
            <span>Ordered stations with paired supply and return distribution segments.</span>
          </button>
        </div>
      </section>
    );
  }

  return (
    <section className="editor-page network-page">
      <div className="page-heading network-heading">
        <div>
          <span className="eyebrow">Topology Editor</span>
          <h1>{`${type === "one_pipe" ? "One" : "Two"}-Pipe Network`}</h1>
          <p>Edit the physical order while previewing the user-defined distribution layout.</p>
        </div>
        <button
          type="button"
          className="button ghost"
          onClick={() => {
            if (window.confirm("Replace the current network with a new topology?")) {
              const next = { ...document };
              delete next.network;
              onChange(next);
            }
          }}
        >
          Change Topology
        </button>
      </div>
      <CompactNetworkEditor document={document} onChange={onChange} />
    </section>
  );
}
