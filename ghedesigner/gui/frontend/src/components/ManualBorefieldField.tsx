import type { FieldProps } from "@rjsf/utils";

import { isJsonObject } from "../types";
import { CoordinateRows, type Coordinate } from "./BoundaryFields";

export const pairBoreholeCoordinates = (xValue: unknown, yValue: unknown): Coordinate[] => {
  const xCoordinates = Array.isArray(xValue) ? xValue : [];
  const yCoordinates = Array.isArray(yValue) ? yValue : [];
  return Array.from({ length: Math.max(xCoordinates.length, yCoordinates.length) }, (_, index) => [
    typeof xCoordinates[index] === "number" ? xCoordinates[index] : 0,
    typeof yCoordinates[index] === "number" ? yCoordinates[index] : 0,
  ]);
};

export const splitBoreholeCoordinates = (coordinates: Coordinate[]) => ({
  x: coordinates.map(([x]) => x),
  y: coordinates.map(([, y]) => y),
});

export function ManualBorefieldField({ fieldPathId, formData, onChange, required }: FieldProps) {
  const data = isJsonObject(formData) ? formData : {};
  const coordinates = pairBoreholeCoordinates(data.x, data.y);
  const activeLength = typeof data.H === "number" ? data.H : "";
  const idPrefix = fieldPathId.name ?? "manual-borefield";

  return (
    <fieldset className="boundary-field manual-borefield-field">
      <legend>Pre-Designed Borefield{required ? " *" : ""}</legend>
      <div className="manual-borefield-settings">
        <label htmlFor={`${idPrefix}-arrangement`}>
          Borefield Arrangement
          <select
            id={`${idPrefix}-arrangement`}
            value="MANUAL"
            onChange={(event) => {
              if (event.currentTarget.value === "RECTANGLE") {
                onChange(
                  { arrangement: "RECTANGLE", ...(typeof data.H === "number" ? { H: data.H } : {}) },
                  fieldPathId.path,
                );
              }
            }}
          >
            <option value="MANUAL">Manual Borefield Coordinates</option>
            <option value="RECTANGLE">Rectangular Borefield</option>
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
                onChange({ ...data, H: event.currentTarget.valueAsNumber }, fieldPathId.path);
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
            onChange({ ...data, arrangement: "MANUAL", ...splitBoreholeCoordinates(next) }, fieldPathId.path);
          }}
        />
      </div>
    </fieldset>
  );
}
