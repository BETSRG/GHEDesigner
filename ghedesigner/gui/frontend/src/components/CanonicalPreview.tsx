import { useEffect, useMemo, useState } from "react";
import { Background, Controls, MarkerType, ReactFlow, type Edge, type Node } from "@xyflow/react";

import { api } from "../api";
import type { InputDocument, NetworkPreview } from "../types";

const branchColor: Record<string, string> = {
  pipe: "#547d78",
  building: "#cf6f47",
  ground_heat_exchanger: "#2f8177",
  source_sink_heat_exchanger: "#7d66a6",
  pump: "#d3a12d",
  bypass: "#9aabb0",
};

const flowElements = (preview: NetworkPreview): { nodes: Node[]; edges: Edge[] } => {
  const nodes = preview.nodes.map((node, index) => ({
    id: node.id,
    position: { x: (index % 5) * 180, y: Math.floor(index / 5) * 120 },
    data: { label: node.id.replace(/^__/, "") },
    className: "preview-node",
  }));
  const edges = preview.branches.map((branch) => ({
    id: branch.id,
    source: branch.node_a,
    target: branch.node_b,
    label: branch.component ?? branch.id.replace(/^__/, ""),
    type: "smoothstep",
    markerEnd: { type: MarkerType.ArrowClosed, color: branchColor[branch.type] ?? "#547d78" },
    style: {
      stroke: branchColor[branch.type] ?? "#547d78",
      strokeWidth: 2,
    },
  }));
  return { nodes, edges };
};

export function CanonicalPreview({ document }: { document: InputDocument }) {
  const [preview, setPreview] = useState<NetworkPreview | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!document.network) {
      setPreview(null);
      return;
    }
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      api
        .compileNetwork(document, controller.signal)
        .then((result) => {
          setPreview(result);
          setError(null);
        })
        .catch((caught: unknown) => {
          if (controller.signal.aborted) return;
          setPreview(null);
          const message = caught instanceof Error ? caught.message : "Unable to compile the network.";
          api.clientLog("error", "network_preview_failed", { message });
          setError(message);
        });
    }, 250);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [document]);

  const elements = useMemo(() => (preview ? flowElements(preview) : { nodes: [], edges: [] }), [preview]);

  return (
    <div className="canonical-preview">
      <div className="panel-heading compact-heading">
        <div>
          <span className="eyebrow">Solver Projection</span>
          <h2>Canonical Network Preview</h2>
        </div>
      </div>
      {error ? (
        <div className="canvas-message error">{error}</div>
      ) : preview ? (
        <div className="network-canvas preview-canvas">
          <ReactFlow nodes={elements.nodes} edges={elements.edges} fitView nodesDraggable={false} nodesConnectable={false}>
            <Background gap={24} size={1} />
            <Controls showInteractive={false} />
          </ReactFlow>
        </div>
      ) : (
        <div className="canvas-message">Complete the network fields to generate a preview.</div>
      )}
    </div>
  );
}
