import { describe, expect, it } from "vitest";

import { createCompactNetwork, rebuildSegments, updateCompactStations } from "./networkUtils";
import type { InputDocument } from "./types";

const document: InputDocument = {
  version: 4,
  building: { building_1: {} },
  ground_heat_exchanger: { ghe_1: {} },
};

describe("compact network editing", () => {
  it("builds a closed one-pipe ring", () => {
    const network = createCompactNetwork(document, "one_pipe");

    expect(network.stations).toEqual([{ component: "building_1" }, { component: "ghe_1" }]);
    expect(network.segments).toHaveLength(2);
    expect(network.component_pumps).toEqual({ building_1: "pump_building_1" });
    expect(network).not.toHaveProperty("passive_components");
    expect(network).not.toHaveProperty("bypass_components");
  });

  it("preserves segment properties when station order is unchanged", () => {
    const network = rebuildSegments({
      type: "two_pipe",
      stations: [{ component: "building_1" }, { component: "ghe_1" }],
      segments: [{ id: "supply", from: "building_1", to: "ghe_1", length: 44, diameter: 0.08 }],
    });

    expect(network.segments).toEqual([
      { id: "supply", from: "building_1", to: "ghe_1", length: 44, diameter: 0.08 },
    ]);
  });

  it("removes controls for stations removed from the compact network", () => {
    const withNetwork = { ...document, network: createCompactNetwork(document, "one_pipe") };
    const updated = updateCompactStations(withNetwork, ["ghe_1"]);

    expect(updated.network?.component_pumps).toEqual({});
    expect(updated.network?.stations).toEqual([{ component: "ghe_1" }]);
  });
});
