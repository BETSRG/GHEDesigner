import { describe, expect, it } from "vitest";

import { blankDocument, deepClone } from "./types";
import {
  applyWorkflow,
  collectionFields,
  inferWorkflow,
  seasonalGroundTemperatureRequired,
  soilFields,
  workflowDefinitions,
  workflowDiagnostics,
  workflowSections,
} from "./workflows";

describe("new input defaults", () => {
  it("starts with standalone sizing controls", () => {
    expect(blankDocument().simulation_control).toMatchObject({
      sizing_years: 20,
    });
  });

  it("does not throw when an optional section is absent", () => {
    expect(deepClone(undefined)).toBeUndefined();
  });
});

describe("workflow modes", () => {
  it("lists G-function generation before GHE design", () => {
    expect(workflowDefinitions.slice(0, 2).map(({ id }) => id)).toEqual(["g_function", "standalone_design"]);
  });

  it("infers a pre-designed network as a district simulation", () => {
    expect(
      inferWorkflow({
        version: 4,
        network: { type: "two_pipe" },
        ground_heat_exchanger: { ghe: { pre_designed: { arrangement: "MANUAL" } } },
      }),
    ).toBe("district_simulation");
  });

  it("retains an unfinished district workflow before a topology is selected", () => {
    expect(inferWorkflow(applyWorkflow(blankDocument(), "district_design"))).toBe("district_design");
    expect(inferWorkflow(applyWorkflow(blankDocument(), "district_simulation"))).toBe("district_simulation");
  });

  it("sets mode-specific controls without storing GUI metadata in the JSON", () => {
    const document = applyWorkflow(blankDocument(), "district_simulation");

    expect(document.simulation_control).toMatchObject({
      simulation_years: 20,
      search_method: "SIMULATION_ONLY",
    });
    expect(document).not.toHaveProperty("workflow_mode");
    expect(document.simulation_control).not.toHaveProperty("sizing_years");
  });

  it("reveals horizontal component sections only when relevant", () => {
    const document = applyWorkflow(blankDocument(), "district_design");
    document.network = { type: "two_pipe" };
    if (document.simulation_control) document.simulation_control.horizontal_simulation_considered = true;

    expect(workflowSections(document, "district_design")).toContain("horizontal_piping");
    expect(workflowSections(document, "district_design")).not.toEqual(
      expect.arrayContaining(["network_pipe", "circulation_pump", "bypass"]),
    );
    expect(workflowSections(document, "standalone_design")).not.toContain("network");
  });

  it("only exposes per-GHE circulation pumps in district workflows", () => {
    const district = applyWorkflow(blankDocument(), "district_design");
    district.network = { type: "two_pipe" };

    expect(collectionFields("ground_heat_exchanger", blankDocument(), "standalone_design")).not.toContain(
      "circulation_pump",
    );
    expect(collectionFields("ground_heat_exchanger", district, "district_design")).toContain("circulation_pump");
  });

  it("separates design EFT limits from COP evaluation temperatures", () => {
    const districtDesign = applyWorkflow(blankDocument(), "district_design");
    const districtSimulation = applyWorkflow(blankDocument(), "district_simulation");

    expect(collectionFields("building", blankDocument(), "building_design")).not.toEqual(
      expect.arrayContaining(["max_eft", "min_eft", "heating_cop_evaluation_temperature"]),
    );
    expect(collectionFields("building", districtDesign, "district_design")).toEqual(
      expect.arrayContaining([
        "max_eft",
        "min_eft",
        "heating_cop_evaluation_temperature",
        "cooling_cop_evaluation_temperature",
      ]),
    );
    expect(collectionFields("building", districtSimulation, "district_simulation")).not.toEqual(
      expect.arrayContaining(["max_eft", "min_eft"]),
    );
    expect(collectionFields("building", districtSimulation, "district_simulation")).toEqual(
      expect.arrayContaining(["heating_cop_evaluation_temperature", "cooling_cop_evaluation_temperature"]),
    );

    if (districtSimulation.simulation_control) {
      districtSimulation.simulation_control.load_method = "HOURLY";
      districtSimulation.simulation_control.constant_cop = false;
    }
    expect(collectionFields("building", districtSimulation, "district_simulation")).not.toEqual(
      expect.arrayContaining(["heating_cop_evaluation_temperature", "cooling_cop_evaluation_temperature"]),
    );
  });

  it("removes design EFT limits when switching to district simulation", () => {
    const document = applyWorkflow(blankDocument(), "district_design");
    document.building = { building_1: { max_eft: 35, min_eft: 5 } };

    const simulation = applyWorkflow(document, "district_simulation");

    expect(simulation.building?.building_1).not.toHaveProperty("max_eft");
    expect(simulation.building?.building_1).not.toHaveProperty("min_eft");
  });

  it("removes unused per-GHE circulation pumps when leaving a district workflow", () => {
    const district = applyWorkflow(blankDocument(), "district_design");
    district.ground_heat_exchanger = { ghe_1: { circulation_pump: { reference_pressure_drop: 1000 } } };

    expect(applyWorkflow(district, "standalone_design").ground_heat_exchanger).toEqual({ ghe_1: {} });
  });

  it("shows the seasonal soil model last and only when horizontal simulation requires it", () => {
    const document = applyWorkflow(blankDocument(), "district_simulation");

    expect(seasonalGroundTemperatureRequired(document)).toBe(false);
    expect(soilFields(document)).toEqual(["conductivity", "rho_cp", "undisturbed_temp"]);

    document.horizontal_piping = { buried_pipe: {} };
    expect(seasonalGroundTemperatureRequired(document)).toBe(false);
    if (document.simulation_control) document.simulation_control.horizontal_simulation_considered = true;

    expect(seasonalGroundTemperatureRequired(document)).toBe(true);
    expect(soilFields(document)).toEqual([
      "conductivity",
      "rho_cp",
      "undisturbed_temp",
      "ground_temperature_model",
    ]);
    expect(workflowDiagnostics(document, "district_simulation")).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ pointer: "/soil/ground_temperature_model" }),
      ]),
    );
  });

  it("does not report an empty district draft as runnable", () => {
    const document = applyWorkflow(blankDocument(), "district_design");

    expect(workflowDiagnostics(document, "district_design").map((issue) => issue.pointer)).toEqual(
      expect.arrayContaining(["/ground_heat_exchanger", "/building", "/network"]),
    );
  });
});
