import { describe, expect, it } from "vitest";

import { createCompactNetwork, rebuildSegments, updateCompactStations } from "./networkUtils";
import type { InputDocument } from "./types";

const document: InputDocument = {
  schema_version: 3,
  buildings: { building_1: {} },
  ground_heat_exchangers: { ghe_1: {} },
};

describe("compact network editing", () => {
  it("builds a closed one-pipe ring", () => {
    const network = createCompactNetwork(document, "one_pipe");

    expect(network.stations).toEqual([{ component_id: "building_1" }, { component_id: "ghe_1" }]);
    expect(network.segments).toHaveLength(2);
    expect(network.component_pumps).toEqual({ building_1: "pump_building_1" });
    expect(network).not.toHaveProperty("distribution_pipe_defaults");
    expect(network.segments).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          diameter_m: 0.1,
          surface_roughness_m: 0.000001,
          minor_loss_coefficient: 0,
        }),
      ]),
    );
    expect(network).not.toHaveProperty("passive_components");
    expect(network).not.toHaveProperty("bypass_components");
  });

  it("preserves segment properties when station order is unchanged", () => {
    const network = rebuildSegments({
      type: "two_pipe",
      stations: [{ component_id: "building_1" }, { component_id: "ghe_1" }],
      segments: [
        {
          id: "supply",
          from_component_id: "building_1",
          to_component_id: "ghe_1",
          length_m: 44,
          diameter_m: 0.08,
        },
      ],
    });

    expect(network.segments).toEqual([
      {
        id: "supply",
        from_component_id: "building_1",
        to_component_id: "ghe_1",
        length_m: 44,
        diameter_m: 0.08,
        surface_roughness_m: 0.000001,
        minor_loss_coefficient: 0,
      },
    ]);
  });

  it("removes controls for stations removed from the compact network", () => {
    const withNetwork = { ...document, network: createCompactNetwork(document, "one_pipe") };
    const updated = updateCompactStations(withNetwork, ["ghe_1"]);

    expect(updated.network?.component_pumps).toEqual({});
    expect(updated.network?.stations).toEqual([{ component_id: "ghe_1" }]);
  });
});
