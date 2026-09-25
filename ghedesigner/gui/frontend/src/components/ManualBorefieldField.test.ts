import { describe, expect, it } from "vitest";

import { pairBoreholeCoordinates, splitBoreholeCoordinates } from "./ManualBorefieldField";

describe("manual borefield coordinates", () => {
  it("pairs the separate schema arrays into editable borehole rows", () => {
    expect(pairBoreholeCoordinates([1, 2], [3, 4])).toEqual([
      [1, 3],
      [2, 4],
    ]);
  });

  it("normalizes mismatched coordinate arrays without dropping a borehole", () => {
    expect(pairBoreholeCoordinates([1, 2], [3])).toEqual([
      [1, 3],
      [2, 0],
    ]);
  });

  it("writes paired rows back to the canonical x and y arrays", () => {
    expect(
      splitBoreholeCoordinates([
        [1, 3],
        [2, 4],
      ]),
    ).toEqual({ x: [1, 2], y: [3, 4] });
  });
});
