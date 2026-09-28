import type { FieldProps } from "@rjsf/utils";

import { isJsonObject } from "../types";
import { CoordinateRows, type Coordinate } from "./BoundaryFields";

export const boreholeCoordinates = (value: unknown): Coordinate[] => {
  const coordinates = Array.isArray(value) ? value : [];
  return coordinates.map((coordinate) => {
    if (!isJsonObject(coordinate)) return [0, 0];
    return [
      typeof coordinate.x === "number" ? coordinate.x : 0,
      typeof coordinate.y === "number" ? coordinate.y : 0,
    ];
  });
};

export const serializeBoreholeCoordinates = (coordinates: Coordinate[]) =>
  coordinates.map(([x, y]) => ({ x, y }));

export function ManualBorefieldField({ fieldPathId, formData, onChange, required }: FieldProps) {
  const data = isJsonObject(formData) ? formData : {};
  const coordinates = boreholeCoordinates(data.borehole_coordinates_m);
  const activeLength = typeof data.active_borehole_length_m === "number" ? data.active_borehole_length_m : "";
  const idPrefix = fieldPathId.name ?? "manual-borefield";

  return (
    <fieldset className="boundary-field manual-borefield-field">
      <legend>Pre-Designed Borefield{required ? " *" : ""}</legend>
      <div className="manual-borefield-settings">
        <label htmlFor={`${idPrefix}-arrangement`}>
          Borefield Arrangement
          <select
            id={`${idPrefix}-arrangement`}
            value="manual"
            onChange={(event) => {
              if (event.currentTarget.value === "rectangle") {
                onChange(
                  {
                    arrangement: "rectangle",
                    ...(typeof data.active_borehole_length_m === "number"
                      ? { active_borehole_length_m: data.active_borehole_length_m }
                      : {}),
                  },
                  fieldPathId.path,
                );
              }
            }}
          >
            <option value="manual">Manual Borefield Coordinates</option>
            <option value="rectangle">Rectangular Borefield</option>
          </select>
        </label>
        <label htmlFor={`${idPrefix}-active-length`}>
          Borehole Active Length (m) *
          <input
            id={`${idPrefix}-active-length`}
            type="number"
            min={0}
            step="any"
            value={activeLength}
            onChange={(event) => {
              if (Number.isFinite(event.currentTarget.valueAsNumber)) {
                onChange(
                  { ...data, active_borehole_length_m: event.currentTarget.valueAsNumber },
                  fieldPathId.path,
                );
              }
            }}
          />
        </label>
      </div>
      <div className="manual-borefield-coordinates">
        <h4>Borehole Coordinates</h4>
        <p className="field-description">
          Enter one paired X and Y coordinate for each borehole. Coordinates are measured in meters.
        </p>
        <CoordinateRows
          idPrefix={`${idPrefix}-coordinates`}
          points={coordinates}
          itemName="Borehole"
          addLabel="Add Borehole"
          removeLabel="Remove Borehole"
          emptyMessage="No borehole coordinates are defined."
          onChange={(next) => {
            onChange(
              { ...data, arrangement: "manual", borehole_coordinates_m: serializeBoreholeCoordinates(next) },
              fieldPathId.path,
            );
          }}
        />
      </div>
    </fieldset>
  );
}
