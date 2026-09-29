import { ArrowDown, ArrowUp, Plus, Trash2 } from "lucide-react";

import { allComponentIds, rebuildSegments, updateCompactStations } from "../networkUtils";
import type { InputDocument, JsonObject } from "../types";
import { deepClone, isJsonObject } from "../types";
import guiDefaults from "../inputDefaults.schema-v3.json";
import { CanonicalPreview } from "./CanonicalPreview";
import { ObjectJsonEditor } from "./ObjectJsonEditor";

interface CompactNetworkEditorProps {
  document: InputDocument;
  onChange: (document: InputDocument) => void;
  onApply: (document: InputDocument) => void;
}

const stationIds = (network: JsonObject) =>
  Array.isArray(network.stations)
    ? network.stations
        .filter(isJsonObject)
        .map((station) => station.component_id)
        .filter((id): id is string => typeof id === "string")
    : [];

export function CompactNetworkEditor({ document, onChange, onApply }: CompactNetworkEditorProps) {
  const network = document.network ?? {};
  const stations = stationIds(network);
  const components = allComponentIds(document);
  const available = components.filter((id) => !stations.includes(id));
  const segments = Array.isArray(network.segments) ? network.segments.filter(isJsonObject) : [];

  const setNetworkField = (field: string, value: unknown) => {
    const next = deepClone(document);
    next.network = { ...(next.network ?? {}), [field]: value as never };
    onChange(next);
  };

  const setStations = (nextStations: string[]) => onChange(updateCompactStations(document, nextStations));

  const moveStation = (index: number, offset: number) => {
    const target = index + offset;
    if (target < 0 || target >= stations.length) return;
    const next = [...stations];
    [next[index], next[target]] = [next[target], next[index]];
    setStations(next);
  };

  const addStation = () => {
    if (available[0]) setStations([...stations, available[0]]);
  };

  const replaceSegment = (index: number, value: JsonObject) => {
    const next = deepClone(document);
    const nextNetwork = next.network ?? {};
    const nextSegments = Array.isArray(nextNetwork.segments) ? [...nextNetwork.segments] : [];
    nextSegments[index] = value;
    nextNetwork.segments = nextSegments;
    next.network = nextNetwork;
    onChange(next);
  };

  const massControl = isJsonObject(network.mass_flow_control) ? network.mass_flow_control : {};

  return (
    <div className="network-editor-grid">
      <div className="network-controls-column">
        {network.type === "one_pipe" && (
          <div className="form-surface">
            <div className="field-grid two-column inset-fields">
              <label>
                Distribution-Flow Multiplier
                <input
                  type="number"
                  min="0"
                  step="0.05"
                  value={
                    typeof massControl.distribution_flow_multiplier === "number"
                      ? massControl.distribution_flow_multiplier
                      : guiDefaults.network.one_pipe_mass_flow_control.distribution_flow_multiplier
                  }
                  onChange={(event) =>
                    setNetworkField("mass_flow_control", {
                      ...massControl,
                      distribution_flow_multiplier: Number(event.target.value),
                    })
                  }
                />
              </label>
              <label>
                Minimum Circulation [kg/s]
                <input
                  type="number"
                  min="0"
                  step="0.01"
                  value={
                    typeof massControl.minimum_distribution_mass_flow_rate_kg_per_s === "number"
                      ? massControl.minimum_distribution_mass_flow_rate_kg_per_s
                      : guiDefaults.network.one_pipe_mass_flow_control.minimum_distribution_mass_flow_rate_kg_per_s
                  }
                  onChange={(event) =>
                    setNetworkField("mass_flow_control", {
                      ...massControl,
                      minimum_distribution_mass_flow_rate_kg_per_s: Number(event.target.value),
                    })
                  }
                />
              </label>
            </div>
          </div>
        )}

        <section className="form-surface">
          <div className="panel-heading compact-heading">
            <div>
              <span className="eyebrow">Physical Order</span>
              <h2>Stations</h2>
            </div>
            {available.length ? (
              <button type="button" className="button secondary compact" onClick={addStation}>
                <Plus size={15} /> Add Station
              </button>
            ) : (
              <span className="station-assignment-status">
                {components.length ? "All Components Assigned" : "No Components Available"}
              </span>
            )}
          </div>
          <p className="station-help">
            Stations reference existing buildings, GHEs, or heat exchangers. Each physical component can appear only once.
            {!available.length && components.length
              ? " Add another physical component before extending the network."
              : ""}
          </p>
          <div className="station-list">
            {stations.map((id, index) => (
              <div className="station-row" key={id}>
                <span className="station-index">{index + 1}</span>
                <select
                  value={id}
                  onChange={(event) => {
                    const next = [...stations];
                    next[index] = event.target.value;
                    setStations(next);
                  }}
                >
                  <option value={id}>{id}</option>
                  {available.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
                <button type="button" className="icon-button" onClick={() => moveStation(index, -1)} disabled={index === 0}>
                  <ArrowUp size={15} />
                </button>
                <button
                  type="button"
                  className="icon-button"
                  onClick={() => moveStation(index, 1)}
                  disabled={index === stations.length - 1}
                >
                  <ArrowDown size={15} />
                </button>
                <button type="button" className="icon-button danger" onClick={() => setStations(stations.filter((_, i) => i !== index))}>
                  <Trash2 size={15} />
                </button>
              </div>
            ))}
            {!stations.length && <div className="empty-inline">Add components before assembling stations.</div>}
          </div>
        </section>

        <section className="form-surface">
          <div className="panel-heading compact-heading">
            <div>
              <span className="eyebrow">Consecutive Connections</span>
              <h2>Distribution Segments</h2>
            </div>
          </div>
          <div className="segment-list">
            {segments.map((segment, index) => (
              <div className="segment-row" key={String(segment.id)}>
                <div>
                  <strong>{String(segment.from_component_id)} → {String(segment.to_component_id)}</strong>
                  <small>{String(segment.id)}</small>
                </div>
                <label>
                  Length [m]
                  <input
                    type="number"
                    min="0"
                    step="0.1"
                    value={
                      typeof segment.length_m === "number"
                        ? segment.length_m
                        : guiDefaults.network.segment.length_m
                    }
                    onChange={(event) => replaceSegment(index, { ...segment, length_m: Number(event.target.value) })}
                  />
                </label>
                {typeof segment.thermal_model_id === "string" ? (
                  <div>
                    <strong>Hydraulic pipe</strong>
                    <small>Uses {segment.thermal_model_id} pipe diameter and roughness.</small>
                  </div>
                ) : (
                  <>
                    <label>
                      Diameter [m]
                      <input
                        type="number"
                        min="0"
                        step="0.001"
                        value={
                          typeof segment.diameter_m === "number"
                            ? segment.diameter_m
                            : guiDefaults.network.segment.diameter_m
                        }
                        onChange={(event) =>
                          replaceSegment(index, { ...segment, diameter_m: Number(event.target.value) })
                        }
                      />
                    </label>
                    <label>
                      Surface Roughness [m]
                      <input
                        type="number"
                        min="0"
                        step="0.000001"
                        value={
                          typeof segment.surface_roughness_m === "number"
                            ? segment.surface_roughness_m
                            : guiDefaults.network.segment.surface_roughness_m
                        }
                        onChange={(event) =>
                          replaceSegment(index, { ...segment, surface_roughness_m: Number(event.target.value) })
                        }
                      />
                    </label>
                  </>
                )}
                <label>
                  Minor-Loss Coefficient [-]
                  <input
                    type="number"
                    min="0"
                    step="0.1"
                    value={
                      typeof segment.minor_loss_coefficient === "number"
                        ? segment.minor_loss_coefficient
                        : guiDefaults.network.segment.minor_loss_coefficient
                    }
                    onChange={(event) =>
                      replaceSegment(index, { ...segment, minor_loss_coefficient: Number(event.target.value) })
                    }
                  />
                </label>
              </div>
            ))}
          </div>
        </section>

        <ObjectJsonEditor
          label="Advanced Compact-Network JSON"
          value={network}
          onApply={(value) => onApply({ ...document, network: rebuildSegments(value) })}
        />
      </div>
      <CanonicalPreview document={document} />
    </div>
  );
}
