import { ArrowDown, ArrowUp, CornerDownLeft } from "lucide-react";

import { componentType } from "../networkUtils";
import type { InputDocument, JsonObject } from "../types";
import { isJsonObject } from "../types";

interface DisplayStation {
  id: string;
  type: string;
}

const componentTypeLabel: Record<string, string> = {
  building: "Building",
  ground_heat_exchanger: "GHE",
  source_sink_heat_exchanger: "Heat Exchanger",
};

const stationIds = (network: JsonObject): string[] =>
  Array.isArray(network.stations)
    ? network.stations
        .filter(isJsonObject)
        .map((station) => station.component)
        .filter((id): id is string => typeof id === "string")
    : [];

const networkSegments = (network: JsonObject): JsonObject[] =>
  Array.isArray(network.segments) ? network.segments.filter(isJsonObject) : [];

const segmentLength = (segment: JsonObject | undefined): string => {
  if (!segment || typeof segment.length !== "number") return "Length Not Set";
  return `${segment.length.toLocaleString(undefined, { maximumFractionDigits: 2 })} m`;
};

export function CanonicalPreview({ document }: { document: InputDocument }) {
  const network = document.network ?? {};
  const type = network.type === "two_pipe" ? "two_pipe" : "one_pipe";
  const ids = stationIds(network);
  const segments = networkSegments(network);
  const stations: DisplayStation[] = ids.map((id) => ({
    id,
    type: componentType(document, id) ?? "unknown",
  }));

  const segmentBetween = (index: number, closeLoop = false) => {
    const from = ids[index];
    const to = closeLoop ? ids[0] : ids[index + 1];
    return segments.find((segment) => segment.from === from && segment.to === to);
  };

  return (
    <div className="canonical-preview">
      <div className="panel-heading compact-heading">
        <div>
          <span className="eyebrow">Topology Summary</span>
          <h2>Network Layout</h2>
        </div>
        <span className="network-type-chip">{type === "one_pipe" ? "One-Pipe Loop" : "Two-Pipe Network"}</span>
      </div>
      <p className="network-preview-help">
        Shows physical components and distribution order. Solver-only nodes, bypasses, and pump branches are hidden.
      </p>
      {stations.length ? (
        <div className="topology-summary">
          <div className={`topology-direction ${type}`}>
            <span>
              <ArrowDown size={14} /> {type === "one_pipe" ? "Loop Flow" : "Supply Flow"}
            </span>
            {type === "two_pipe" && (
              <span>
                <ArrowUp size={14} /> Return Flow
              </span>
            )}
          </div>
          <ol className="topology-stations">
            {stations.map((station, index) => {
              const nextSegment = index < stations.length - 1 ? segmentBetween(index) : undefined;
              return (
                <li key={station.id}>
                  <div className={`topology-station ${station.type}`}>
                    <span className="topology-station-number">{index + 1}</span>
                    <div>
                      <strong>{station.id}</strong>
                      <small>{componentTypeLabel[station.type] ?? "Physical Component"}</small>
                    </div>
                  </div>
                  {index < stations.length - 1 && (
                    <div
                      className="topology-connection"
                      title={typeof nextSegment?.id === "string" ? nextSegment.id : undefined}
                    >
                      <span className="topology-connection-line" />
                      <small>{segmentLength(nextSegment)}</small>
                    </div>
                  )}
                </li>
              );
            })}
          </ol>
          {type === "one_pipe" ? (
            <div className="topology-loop-return">
              <CornerDownLeft size={17} />
              <span>
                Return To Station 1
                <small>{segmentLength(segmentBetween(stations.length - 1, true))}</small>
              </span>
            </div>
          ) : (
            <p className="topology-return-note">The return line follows the same connections in reverse.</p>
          )}
        </div>
      ) : (
        <div className="canvas-message">Add physical components to generate a network layout.</div>
      )}
    </div>
  );
}
