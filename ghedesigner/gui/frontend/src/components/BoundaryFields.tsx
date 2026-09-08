import type { FieldProps } from "@rjsf/utils";
import { Plus, Trash2 } from "lucide-react";

export type Coordinate = [number, number];

const asCoordinate = (value: unknown): Coordinate => {
  const values = Array.isArray(value) ? value : [];
  return [typeof values[0] === "number" ? values[0] : 0, typeof values[1] === "number" ? values[1] : 0];
};

const asBoundary = (value: unknown): Coordinate[] => (Array.isArray(value) ? value.map(asCoordinate) : []);

interface CoordinateRowsProps {
  idPrefix: string;
  points: Coordinate[];
  onChange: (points: Coordinate[]) => void;
  itemName?: string;
  addLabel?: string;
  removeLabel?: string;
  emptyMessage?: string;
  minimum?: number;
}

export function CoordinateRows({
  idPrefix,
  points,
  onChange,
  itemName = "Point",
  addLabel = "Add Point",
  removeLabel = "Remove Point",
  emptyMessage = "No boundary points are defined.",
  minimum,
}: CoordinateRowsProps) {
  const updateCoordinate = (pointIndex: number, coordinateIndex: 0 | 1, value: number) => {
    const next = points.map((point) => [...point] as Coordinate);
    next[pointIndex][coordinateIndex] = value;
    onChange(next);
  };

  return (
    <div className="boundary-points">
      {points.map((point, pointIndex) => (
        <div className="boundary-point" key={`${idPrefix}-${pointIndex}`}>
          <strong>{itemName} {pointIndex + 1}</strong>
          <label htmlFor={`${idPrefix}-${pointIndex}-x`}>
            X Coordinate (m)
            <input
              id={`${idPrefix}-${pointIndex}-x`}
              type="number"
              min={minimum}
              step="any"
              value={point[0]}
              onChange={(event) => {
                if (Number.isFinite(event.currentTarget.valueAsNumber)) {
                  updateCoordinate(pointIndex, 0, event.currentTarget.valueAsNumber);
                }
              }}
            />
          </label>
          <label htmlFor={`${idPrefix}-${pointIndex}-y`}>
            Y Coordinate (m)
            <input
              id={`${idPrefix}-${pointIndex}-y`}
              type="number"
              min={minimum}
              step="any"
              value={point[1]}
              onChange={(event) => {
                if (Number.isFinite(event.currentTarget.valueAsNumber)) {
                  updateCoordinate(pointIndex, 1, event.currentTarget.valueAsNumber);
                }
              }}
            />
          </label>
          <button
            type="button"
            className="button danger compact boundary-remove"
            aria-label={`${removeLabel} ${pointIndex + 1}`}
            onClick={() => onChange(points.filter((_, index) => index !== pointIndex))}
          >
            <Trash2 size={14} /> {removeLabel}
          </button>
        </div>
      ))}
      {points.length === 0 && <p className="empty-inline">{emptyMessage}</p>}
      <button
        type="button"
        className="button secondary compact boundary-add"
        onClick={() => onChange([...points, [0, 0]])}
      >
        <Plus size={14} /> {addLabel}
      </button>
    </div>
  );
}

export function PropertyBoundaryField({ fieldPathId, formData, onChange, required, schema }: FieldProps) {
  const points = asBoundary(formData);
  const idPrefix = fieldPathId.name ?? "property-boundary";

  return (
    <fieldset className="boundary-field">
      <legend>Go Zone Boundary{required ? " *" : ""}</legend>
      {schema.description && <p className="field-description">{schema.description}</p>}
      <CoordinateRows
        idPrefix={idPrefix}
        points={points}
        minimum={0}
        onChange={(next) => onChange(next, fieldPathId.path)}
      />
    </fieldset>
  );
}

export function NoGoBoundariesField({ fieldPathId, formData, onChange, required, schema }: FieldProps) {
  const boundaries = Array.isArray(formData) ? formData.map(asBoundary) : [];
  const idPrefix = fieldPathId.name ?? "no-go-boundaries";

  const updateBoundary = (boundaryIndex: number, points: Coordinate[]) => {
    const next = boundaries.map((boundary) => boundary.map((point) => [...point] as Coordinate));
    next[boundaryIndex] = points;
    onChange(next, fieldPathId.path);
  };

  return (
    <fieldset className="boundary-field no-go-boundary-field">
      <legend>No-Go Zone Boundaries{required ? " *" : ""}</legend>
      {schema.description && <p className="field-description">{schema.description}</p>}
      <div className="no-go-zones">
        {boundaries.map((boundary, boundaryIndex) => (
          <section className="no-go-zone" key={`${idPrefix}-${boundaryIndex}`}>
            <div className="boundary-zone-heading">
              <h4>No-Go Zone {boundaryIndex + 1}</h4>
              <button
                type="button"
                className="button danger compact"
                aria-label={`Remove No-Go Zone ${boundaryIndex + 1}`}
                onClick={() => onChange(boundaries.filter((_, index) => index !== boundaryIndex), fieldPathId.path)}
              >
                <Trash2 size={14} /> Remove Zone
              </button>
            </div>
            <CoordinateRows
              idPrefix={`${idPrefix}-${boundaryIndex}`}
              points={boundary}
              minimum={0}
              onChange={(next) => updateBoundary(boundaryIndex, next)}
            />
          </section>
        ))}
        {boundaries.length === 0 && <p className="empty-inline">No no-go zones are defined.</p>}
        <button
          type="button"
          className="button secondary compact boundary-add"
          onClick={() => onChange([...boundaries, []], fieldPathId.path)}
        >
          <Plus size={14} /> Add No-Go Zone
        </button>
      </div>
    </fieldset>
  );
}
