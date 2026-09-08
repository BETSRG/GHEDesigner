import { useEffect, useState } from "react";

import type { JsonObject } from "../types";
import { isJsonObject } from "../types";

interface ObjectJsonEditorProps {
  value: JsonObject;
  onApply: (value: JsonObject) => void;
  label: string;
}

export function ObjectJsonEditor({ value, onApply, label }: ObjectJsonEditorProps) {
  const [text, setText] = useState(() => JSON.stringify(value, null, 2));
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setText(JSON.stringify(value, null, 2));
    setError(null);
  }, [value]);

  const apply = () => {
    try {
      const parsed: unknown = JSON.parse(text);
      if (!isJsonObject(parsed)) throw new Error("The value must be a JSON object.");
      onApply(parsed);
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Invalid JSON.");
    }
  };

  return (
    <div className="object-json-editor">
      <label>{label}</label>
      <textarea value={text} spellCheck={false} onChange={(event) => setText(event.target.value)} />
      <div className="object-json-footer">
        {error ? <span className="inline-error">{error}</span> : <span />}
        <button type="button" className="button secondary compact" onClick={apply}>
          Apply
        </button>
      </div>
    </div>
  );
}
