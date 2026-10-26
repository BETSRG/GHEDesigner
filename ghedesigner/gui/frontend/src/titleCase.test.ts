import { describe, expect, it } from "vitest";

import { toTitleCase } from "./titleCase";

describe("title case", () => {
  it("capitalizes field and section labels while preserving small words", () => {
    expect(toTitleCase("Maximum borehole spacing in the x direction")).toBe(
      "Maximum Borehole Spacing in the X Direction",
    );
    expect(toTitleCase("Source and sink heat exchangers")).toBe("Source and Sink Heat Exchangers");
    expect(toTitleCase("Read a CSV column by name")).toBe("Read a CSV Column by Name");
  });

  it("preserves technical acronyms and hyphenated terms", () => {
    expect(toTitleCase("Heat pump COP")).toBe("Heat Pump COP");
    expect(toTitleCase("GHEDesigner input file")).toBe("GHEDesigner Input File");
    expect(toTitleCase("CSV column name")).toBe("CSV Column Name");
  });
});
