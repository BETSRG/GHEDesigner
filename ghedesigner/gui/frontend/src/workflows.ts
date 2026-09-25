import type { Diagnostic, InputDocument, JsonObject, WorkflowMode } from "./types";
import { deepClone, isJsonObject } from "./types";

export type { WorkflowMode } from "./types";

export interface WorkflowDefinition {
  id: WorkflowMode;
  label: string;
  shortLabel: string;
  description: string;
  outcome: string;
  steps: string[];
}

export const workflowDefinitions: WorkflowDefinition[] = [
  {
    id: "g_function",
    label: "G-Function Calculation",
    shortLabel: "G-Function",
    description: "Calculate response factors for borefield coordinates and depth that are already known.",
    outcome: "Short- and long-time-step g-function data for each pre-designed GHE.",
    steps: ["Define fluid and soil", "Enter the known borefield", "Review and run"],
  },
  {
    id: "standalone_design",
    label: "Standalone GHE Design",
    shortLabel: "GHE Design",
    description: "Select a borefield arrangement and size borehole depth from loads applied directly to each GHE.",
    outcome: "A sized vertical borefield with design summaries, coordinates, loads, and g-functions.",
    steps: ["Define fluid and soil", "Enter sizing duration", "Configure GHE constraints and loads", "Review and run"],
  },
  {
    id: "building_design",
    label: "Building + GHE Design",
    shortLabel: "Building + GHE",
    description: "Convert one building load profile with fixed COP behavior and size one connected GHE.",
    outcome: "A GHE design sized from one building's heating and cooling loads.",
    steps: ["Define fluid and soil", "Enter sizing duration", "Configure one building and one GHE", "Review and run"],
  },
  {
    id: "district_design",
    label: "District-System Sizing",
    shortLabel: "District Sizing",
    description: "Search coordinated GHE layouts and depths while repeatedly evaluating the complete thermal network.",
    outcome: "Selected borefield layouts, sized depths, search history, and a final detailed system simulation.",
    steps: ["Define shared properties", "Configure loads and sizable GHEs", "Build the network", "Choose the search", "Run sizing"],
  },
  {
    id: "district_simulation",
    label: "District-System Simulation",
    shortLabel: "District Simulation",
    description: "Simulate a network whose GHE coordinates and borehole depths are already fixed.",
    outcome: "Detailed component, network, thermal, flow, pressure-loss, and pump time series.",
    steps: ["Define shared properties", "Configure loads and pre-designed GHEs", "Build the network", "Choose fidelity", "Run simulation"],
  },
];

export const workflowDefinition = (mode: WorkflowMode) =>
  workflowDefinitions.find((definition) => definition.id === mode) ?? workflowDefinitions[0];

const gheEntries = (document: InputDocument): JsonObject[] =>
  Object.values(document.ground_heat_exchanger ?? {}).filter(isJsonObject);

const buildingEntries = (document: InputDocument): JsonObject[] =>
  Object.values(document.building ?? {}).filter(isJsonObject);

export const copEvaluationTemperaturesRelevant = (document: InputDocument, mode: WorkflowMode): boolean => {
  if (!mode.startsWith("district_")) return false;
  const controls = document.simulation_control;
  const loadMethod = typeof controls?.load_method === "string" ? controls.load_method.toUpperCase() : "";
  return controls?.constant_cop === true || loadMethod === "HYBRID";
};

export const inferWorkflow = (document: InputDocument): WorkflowMode => {
  const ghes = gheEntries(document);
  const controls = document.simulation_control;
  if (isJsonObject(document.network)) {
    if (ghes.length > 0 && ghes.every((ghe) => isJsonObject(ghe.pre_designed))) return "district_simulation";
    return document.simulation_control?.search_method === "SIMULATION_ONLY"
      ? "district_simulation"
      : "district_design";
  }
  if (typeof controls?.simulation_years === "number") {
    return controls.search_method === "SIMULATION_ONLY" ? "district_simulation" : "district_design";
  }
  if (Object.keys(document.building ?? {}).length > 0) return "building_design";
  if (ghes.length > 0 && ghes.every((ghe) => isJsonObject(ghe.pre_designed))) return "g_function";
  return "standalone_design";
};

const deleteKeys = (object: JsonObject, keys: string[]) => {
  for (const key of keys) delete object[key];
};

export const applyWorkflow = (source: InputDocument, mode: WorkflowMode): InputDocument => {
  const document = deepClone(source);
  document.version ??= 4;
  document.building ??= {};
  document.ground_heat_exchanger ??= {};

  const previousControls = isJsonObject(document.simulation_control) ? document.simulation_control : {};
  const duration =
    typeof previousControls.simulation_years === "number"
      ? previousControls.simulation_years
      : typeof previousControls.sizing_years === "number"
        ? previousControls.sizing_years
        : mode === "standalone_design" || mode === "building_design"
          ? 20
          : 1;

  if (mode === "g_function") {
    delete document.simulation_control;
  } else if (mode === "standalone_design" || mode === "building_design") {
    document.simulation_control = { sizing_years: duration };
  } else {
    document.simulation_control = {
      simulation_years: duration,
      load_method: typeof previousControls.load_method === "string" ? previousControls.load_method : "HYBRID",
      search_method:
        mode === "district_simulation"
          ? "SIMULATION_ONLY"
          : previousControls.search_method === "SIMULATION_ONLY" || typeof previousControls.search_method !== "string"
            ? "GLOBAL_BUPCRS_BR"
            : previousControls.search_method,
      constant_cop: typeof previousControls.constant_cop === "boolean" ? previousControls.constant_cop : true,
      horizontal_segments:
        typeof previousControls.horizontal_segments === "number" ? previousControls.horizontal_segments : 3,
      horizontal_simulation_considered: previousControls.horizontal_simulation_considered === true,
      ...(mode === "district_design"
        ? { exhaustive_search: previousControls.exhaustive_search === true }
        : {}),
    };
  }

  if (mode === "standalone_design" || mode === "g_function" || mode === "building_design") {
    deleteKeys(document, [
      "network",
      "horizontal_piping",
      "source_sink_heat_exchanger",
      "heat_pump",
    ]);
    for (const ghe of gheEntries(document)) delete ghe.circulation_pump;
  }
  if (mode === "standalone_design" || mode === "g_function") document.building = {};
  for (const building of buildingEntries(document)) {
    if (mode !== "district_design") deleteKeys(building, ["max_eft", "min_eft"]);
    if (!copEvaluationTemperaturesRelevant(document, mode)) {
      deleteKeys(building, ["heating_cop_evaluation_temperature", "cooling_cop_evaluation_temperature"]);
    }
  }

  return document;
};

export const workflowSections = (document: InputDocument, mode: WorkflowMode): string[] => {
  const common = ["overview", "fluid", "soil"];
  if (mode === "g_function") return [...common, "ground_heat_exchanger", "review"];
  if (mode === "standalone_design") return [...common, "simulation_control", "ground_heat_exchanger", "review"];
  if (mode === "building_design") {
    return [...common, "simulation_control", "building", "ground_heat_exchanger", "review"];
  }

  const sections = [...common, "simulation_control"];
  if (document.simulation_control?.constant_cop === false) sections.push("heat_pump");
  sections.push("building", "ground_heat_exchanger");
  if (
    document.simulation_control?.horizontal_simulation_considered === true ||
    Object.keys(isJsonObject(document.horizontal_piping) ? document.horizontal_piping : {}).length > 0
  ) sections.push("horizontal_piping");
  if (Object.keys(document.source_sink_heat_exchanger ?? {}).length > 0) sections.push("source_sink_heat_exchanger");
  sections.push("network", "review");
  return sections;
};

export const simulationControlFields = (mode: WorkflowMode): string[] => {
  if (mode === "standalone_design" || mode === "building_design") return ["sizing_years"];
  if (mode === "district_design") {
    return [
      "simulation_years",
      "load_method",
      "search_method",
      "constant_cop",
      "exhaustive_search",
      "horizontal_segments",
      "horizontal_simulation_considered",
    ];
  }
  if (mode === "district_simulation") {
    return [
      "simulation_years",
      "load_method",
      "constant_cop",
      "horizontal_segments",
      "horizontal_simulation_considered",
    ];
  }
  return [];
};

export const seasonalGroundTemperatureRequired = (document: InputDocument): boolean =>
  document.simulation_control?.horizontal_simulation_considered === true &&
  Object.keys(isJsonObject(document.horizontal_piping) ? document.horizontal_piping : {}).length > 0;

export const soilFields = (document: InputDocument): string[] => [
  "conductivity",
  "rho_cp",
  "undisturbed_temp",
  ...(seasonalGroundTemperatureRequired(document) ? ["ground_temperature_model"] : []),
];

export const collectionFields = (
  property: string,
  document: InputDocument,
  mode: WorkflowMode,
): string[] | undefined => {
  if (property === "ground_heat_exchanger") {
    const fields = [
      "flow_rate",
      ...(mode.startsWith("district_") ? ["circulation_pump"] : []),
      "grout",
      "pipe",
      "borehole",
    ];
    if (mode === "g_function" || mode === "district_simulation") fields.push("pre_designed");
    else fields.push("geometric_constraints", "design");
    if (mode === "standalone_design") fields.push("loads");
    return fields;
  }
  if (property === "building") {
    return [
      "heating_load",
      "cooling_load",
      "total_load",
      ...(mode === "district_design" ? ["max_eft", "min_eft"] : []),
      ...(copEvaluationTemperaturesRelevant(document, mode)
        ? ["heating_cop_evaluation_temperature", "cooling_cop_evaluation_temperature"]
        : []),
    ];
  }
  return undefined;
};

const workflowDiagnostic = (message: string, pointer: string, location: string): Diagnostic => ({
  severity: "error",
  message,
  pointer,
  location,
  validator: "workflow",
  suggestions: [],
  source: "semantic",
});

export const workflowDiagnostics = (document: InputDocument, mode: WorkflowMode): Diagnostic[] => {
  const diagnostics: Diagnostic[] = [];
  const ghes = gheEntries(document);
  const buildingCount = Object.keys(document.building ?? {}).length;
  const hasNetwork = isJsonObject(document.network);

  if (ghes.length === 0) {
    diagnostics.push(
      workflowDiagnostic("This workflow requires at least one ground heat exchanger.", "/ground_heat_exchanger", "GHEs"),
    );
  }
  if (mode === "building_design" && buildingCount !== 1) {
    diagnostics.push(
      workflowDiagnostic("Building + GHE design requires exactly one building.", "/building", "Heat pump loads"),
    );
  }
  if (
    seasonalGroundTemperatureRequired(document) &&
    !isJsonObject(isJsonObject(document.soil) ? document.soil.ground_temperature_model : undefined)
  ) {
    diagnostics.push(
      workflowDiagnostic(
        "A seasonal ground temperature model is required when simulating horizontal piping.",
        "/soil/ground_temperature_model",
        "Soil",
      ),
    );
  }
  if (mode === "building_design" && ghes.length !== 1) {
    diagnostics.push(
      workflowDiagnostic("Building + GHE design requires exactly one GHE.", "/ground_heat_exchanger", "GHEs"),
    );
  }
  if (mode.startsWith("district_") && !hasNetwork) {
    diagnostics.push(
      workflowDiagnostic("Choose and configure a district network topology.", "/network", "Network"),
    );
  }
  if (mode.startsWith("district_") && buildingCount === 0) {
    diagnostics.push(
      workflowDiagnostic("A district workflow requires at least one building load component.", "/building", "Heat pump loads"),
    );
  }

  const requiresPredesigned = mode === "g_function" || mode === "district_simulation";
  ghes.forEach((ghe, index) => {
    const isPredesigned = isJsonObject(ghe.pre_designed);
    const isSizable = isJsonObject(ghe.geometric_constraints) && isJsonObject(ghe.design);
    if (requiresPredesigned && !isPredesigned) {
      diagnostics.push(
        workflowDiagnostic(
          "This workflow requires pre-designed borefield coordinates and borehole height for every GHE.",
          "/ground_heat_exchanger",
          `GHE ${index + 1}`,
        ),
      );
    }
    if (!requiresPredesigned && !isSizable) {
      diagnostics.push(
        workflowDiagnostic(
          "This workflow requires geometric constraints and design limits for every GHE.",
          "/ground_heat_exchanger",
          `GHE ${index + 1}`,
        ),
      );
    }
    if (mode === "standalone_design" && !isJsonObject(ghe.loads)) {
      diagnostics.push(
        workflowDiagnostic(
          "Standalone GHE design requires a load source on every GHE.",
          "/ground_heat_exchanger",
          `GHE ${index + 1}`,
        ),
      );
    }
  });

  return diagnostics;
};
