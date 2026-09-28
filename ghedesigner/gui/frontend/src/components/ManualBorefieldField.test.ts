import { describe, expect, it } from "vitest";

import { boreholeCoordinates, serializeBoreholeCoordinates } from "./ManualBorefieldField";

describe("manual borefield coordinates", () => {
  it("converts coordinate objects into editable borehole rows", () => {
    expect(boreholeCoordinates([{ x: 1, y: 3 }, { x: 2, y: 4 }])).toEqual([
      [1, 3],
      [2, 4],
    ]);
  });

  it("normalizes an incomplete coordinate object", () => {
    expect(boreholeCoordinates([{ x: 1 }])).toEqual([[1, 0]]);
  });

  it("writes paired rows back to canonical coordinate objects", () => {
    expect(
      serializeBoreholeCoordinates([
        [1, 3],
        [2, 4],
      ]),
    ).toEqual([{ x: 1, y: 3 }, { x: 2, y: 4 }]);
  });
});
