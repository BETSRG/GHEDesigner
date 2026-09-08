import { useEffect, useState } from "react";
import { Check, RotateCcw } from "lucide-react";

import type { InputDocument, JsonObject } from "../types";
import { isJsonObject } from "../types";

interface JsonEditorProps {
  value: JsonObject;
  onApply: (value: InputDocument) => void;
  title?: string;
}

export function JsonEditor({ value, onApply, title = "Raw JSON" }: JsonEditorProps) {
  const [text, setText] = useState(() => JSON.stringify(value, null, 2));
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setText(JSON.stringify(value, null, 2));
    setError(null);
  }, [value]);

  const apply = () => {
    try {
      const parsed: unknown = JSON.parse(text);
      if (!isJsonObject(parsed)) throw new Error("The document root must be a JSON object.");
      setError(null);
      onApply(parsed as InputDocument);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Invalid JSON.");
    }
  };

  return (
    <section className="json-editor">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Advanced</span>
          <h2>{title}</h2>
        </div>
        <div className="button-row">
          <button type="button" className="button ghost" onClick={() => setText(JSON.stringify(value, null, 2))}>
            <RotateCcw size={15} /> Reset
          </button>
          <button type="button" className="button primary" onClick={apply}>
            <Check size={15} /> Apply JSON
          </button>
        </div>
      </div>
      <textarea
        aria-label={title}
        className="code-editor"
        spellCheck={false}
        value={text}
        onChange={(event) => setText(event.target.value)}
      />
      {error && <div className="inline-error">{error}</div>}
    </section>
  );
}
