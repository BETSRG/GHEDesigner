import type { InputDocument, JsonObject, JsonValue } from "./types";
import { deepClone, isJsonObject } from "./types";

export type CompactNetworkType = "one_pipe" | "two_pipe";

const componentIds = (document: InputDocument) => [
  ...Object.keys(document.buildings ?? {}),
  ...Object.keys(document.ground_heat_exchangers ?? {}),
  ...Object.keys(document.source_sink_heat_exchangers ?? {}),
];

export const componentType = (document: InputDocument, id: string) => {
  if (id in (document.buildings ?? {})) return "building";
  if (id in (document.ground_heat_exchangers ?? {})) return "ground_heat_exchanger";
  if (id in (document.source_sink_heat_exchangers ?? {})) return "source_sink_heat_exchanger";
  return null;
};

const pumpDefinition = (): JsonObject => ({
  wire_to_water_efficiency_fraction: 0.7,
});

export const rebuildSegments = (network: JsonObject): JsonObject => {
  const stations = Array.isArray(network.stations)
    ? network.stations
        .filter(isJsonObject)
        .map((station) => station.component_id)
        .filter((id): id is string => typeof id === "string")
    : [];
  const type = network.type === "one_pipe" ? "one_pipe" : "two_pipe";
  const count = type === "one_pipe" ? stations.length : Math.max(0, stations.length - 1);
  const existing = Array.isArray(network.segments) ? network.segments.filter(isJsonObject) : [];
  const usedIds = new Set<string>();
  const nextId = () => {
    let index = 1;
    while (usedIds.has(`segment_${index}`)) index += 1;
    const id = `segment_${index}`;
    usedIds.add(id);
    return id;
  };
  const segments: JsonValue[] = [];
  for (let index = 0; index < count; index += 1) {
    const from = stations[index];
    const to = stations[(index + 1) % stations.length];
    const previous = existing.find(
      (segment) => segment.from_component_id === from && segment.to_component_id === to,
    );
    const id = typeof previous?.id === "string" && !usedIds.has(previous.id) ? previous.id : nextId();
    usedIds.add(id);
    segments.push({
      ...(previous ?? {}),
      id,
      from_component_id: from,
      to_component_id: to,
      length_m: typeof previous?.length_m === "number" ? previous.length_m : 10,
    });
  }
  return { ...network, segments };
};

export const createCompactNetwork = (document: InputDocument, type: CompactNetworkType): JsonObject => {
  const stations = componentIds(document).map((component_id) => ({ component_id }));
  const pumps: JsonObject = {};
  const componentPumps: JsonObject = {};

  for (const id of componentIds(document)) {
    if (componentType(document, id) === "building") {
      const pumpId = `pump_${id}`;
      pumps[pumpId] = pumpDefinition();
      componentPumps[id] = pumpId;
    }
  }

  const network: JsonObject = {
    type,
    stations,
    distribution_pipe_defaults: { diameter_m: 0.1, surface_roughness_m: 0.000001 },
    segments: [],
    pumps,
    component_pumps: componentPumps,
  };
  if (type === "one_pipe") {
    const pumpId = "distribution_pump";
    pumps[pumpId] = pumpDefinition();
    network.mass_flow_control = {
      distribution_flow_multiplier: 1.5,
      minimum_distribution_mass_flow_rate_kg_per_s: 0.1,
    };
    network.distribution_pump = {
      type: "pump",
      pump_id: pumpId,
    };
  }
  return rebuildSegments(network);
};

export const updateCompactStations = (
  document: InputDocument,
  stationIds: string[],
): InputDocument => {
  const next = deepClone(document);
  const network = isJsonObject(next.network) ? next.network : {};
  network.stations = stationIds.map((component_id) => ({ component_id }));

  const pumps = isJsonObject(network.pumps) ? network.pumps : {};
  const componentPumps = isJsonObject(network.component_pumps) ? network.component_pumps : {};
  const stationSet = new Set(stationIds);

  for (const key of Object.keys(componentPumps)) if (!stationSet.has(key)) delete componentPumps[key];

  for (const id of stationIds) {
    if (componentType(document, id) === "building") {
      if (typeof componentPumps[id] !== "string") {
        let pumpId = `pump_${id}`;
        let suffix = 2;
        while (pumpId in pumps) pumpId = `pump_${id}_${suffix++}`;
        pumps[pumpId] = pumpDefinition();
        componentPumps[id] = pumpId;
      }
    } else {
      delete componentPumps[id];
    }
  }

  network.pumps = pumps;
  network.component_pumps = componentPumps;
  next.network = rebuildSegments(network);
  return next;
};

export const allComponentIds = componentIds;
