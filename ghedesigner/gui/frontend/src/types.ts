export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonObject | JsonValue[];
export type JsonObject = { [key: string]: JsonValue };

export type WorkflowMode =
  | "standalone_design"
  | "g_function"
  | "building_design"
  | "district_design"
  | "district_simulation";

export type InputDocument = JsonObject & {
  version?: number;
  network?: JsonObject;
  building?: JsonObject;
  ground_heat_exchanger?: JsonObject;
  source_sink_heat_exchanger?: JsonObject;
  simulation_control?: JsonObject;
};

export interface Diagnostic {
  severity: "error" | "warning";
  message: string;
  pointer: string;
  location: string;
  validator: string;
  suggestions: string[];
  source: "schema" | "semantic";
}

export interface ValidationResult {
  valid: boolean;
  diagnostics: Diagnostic[];
  error?: string;
}

export interface ExampleSummary {
  name: string;
  label: string;
  network_type: string | null;
  has_buildings: boolean;
  workflow_mode: WorkflowMode;
}

export interface PreviewNode {
  id: string;
}

export interface PreviewBranch {
  id: string;
  type: string;
  node_a: string;
  node_b: string;
  component: string | null;
}

export interface NetworkPreview {
  nodes: PreviewNode[];
  branches: PreviewBranch[];
}

export type SimulationStatus = "queued" | "running" | "cancelling" | "completed" | "failed" | "cancelled";

export interface SimulationJob {
  id: string;
  status: SimulationStatus;
  output_directory: string;
  input_name: string;
  output: string;
  error: string | null;
  return_code: number | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface RunSettings {
  default_output_directory: string;
}

export interface PathSelection {
  path: string | null;
}

export const isJsonObject = (value: unknown): value is JsonObject =>
  typeof value === "object" && value !== null && !Array.isArray(value);

export const deepClone = <T,>(value: T): T =>
  value === undefined ? value : (JSON.parse(JSON.stringify(value)) as T);

export const blankDocument = (): InputDocument => ({
  version: 4,
  fluid: {
    fluid_name: "WATER",
    concentration_percent: 0,
    temperature: 20,
  },
  soil: {
    conductivity: 2.0,
    rho_cp: 2343000,
    undisturbed_temp: 12,
  },
  simulation_control: {
    sizing_years: 20,
  },
  building: {},
  ground_heat_exchanger: {},
});
