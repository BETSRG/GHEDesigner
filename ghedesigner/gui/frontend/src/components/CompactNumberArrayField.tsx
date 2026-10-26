import type { FieldProps } from "@rjsf/utils";
import { useState } from "react";

const formatValues = (value: unknown) =>
  Array.isArray(value) ? value.filter((item): item is number => typeof item === "number").join("\n") : "";

export function CompactNumberArrayField({ fieldPathId, formData, name, onChange, required, schema }: FieldProps) {
  const [text, setText] = useState(() => formatValues(formData));
  const [parseError, setParseError] = useState<string | null>(null);
  const label = schema.title ?? name ?? "Numeric values";
  const inputId = fieldPathId.name ?? name ?? "numeric_values";

  return (
    <div className="form-group compact-array-field">
      <label htmlFor={inputId}>
        {label}{required ? " *" : ""}
      </label>
      {schema.description && <p className="field-description">{schema.description}</p>}
      <textarea
        id={inputId}
        rows={10}
        value={text}
        placeholder="Enter one numeric value per line, or separate values with commas."
        onChange={(event) => {
          const nextText = event.target.value;
          setText(nextText);
          const tokens = nextText.split(/[\s,]+/).filter(Boolean);
          const values = tokens.map(Number);
          if (values.some((value) => !Number.isFinite(value))) {
            setParseError("Every entry must be a finite number.");
            return;
          }
          setParseError(null);
          onChange(values, fieldPathId.path);
        }}
      />
      <div className="compact-array-summary">
        <span>{Array.isArray(formData) ? formData.length : 0} values</span>
        {parseError && <strong>{parseError}</strong>}
      </div>
    </div>
  );
}
