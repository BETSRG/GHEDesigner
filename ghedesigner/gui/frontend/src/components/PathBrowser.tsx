import { FolderOpen, LoaderCircle } from "lucide-react";
import { useState } from "react";

import { api } from "../api";

interface PathBrowserButtonProps {
  kind: "file" | "directory";
  currentPath: string;
  disabled?: boolean;
  onSelect: (path: string) => void;
}

export function PathBrowserButton({ kind, currentPath, disabled, onSelect }: PathBrowserButtonProps) {
  const [browsing, setBrowsing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const target = kind === "file" ? "file" : "directory";

  const browse = async () => {
    setBrowsing(true);
    setError(null);
    try {
      const result = await api.choosePath(kind, currentPath);
      if (result.path) {
        onSelect(result.path);
        api.clientLog("info", "path_selected", { kind });
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : `Unable to browse for a ${target}.`);
    } finally {
      setBrowsing(false);
    }
  };

  return (
    <>
      <button
        type="button"
        className="button secondary path-browse-button"
        disabled={disabled || browsing}
        aria-label={`Browse for ${target}`}
        onClick={() => void browse()}
      >
        {browsing ? <LoaderCircle className="spin" size={16} /> : <FolderOpen size={16} />}
        {browsing ? "Choosing…" : "Browse…"}
      </button>
      {error && <small className="path-browser-error" role="alert">{error}</small>}
    </>
  );
}
