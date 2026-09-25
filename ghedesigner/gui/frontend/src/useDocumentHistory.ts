import { useCallback, useState } from "react";

import { api } from "./api";
import { deepClone, type InputDocument } from "./types";

interface HistoryState {
  past: InputDocument[];
  present: InputDocument;
  future: InputDocument[];
}

export const useDocumentHistory = (initialDocument: InputDocument) => {
  const [state, setState] = useState<HistoryState>({
    past: [],
    present: deepClone(initialDocument),
    future: [],
  });

  const update = useCallback((next: InputDocument | ((current: InputDocument) => InputDocument)) => {
    setState((current) => {
      const document = typeof next === "function" ? next(deepClone(current.present)) : next;
      if (JSON.stringify(document) === JSON.stringify(current.present)) return current;
      const changedRoots = Array.from(new Set([...Object.keys(current.present), ...Object.keys(document)])).filter(
        (key) => JSON.stringify(current.present[key]) !== JSON.stringify(document[key]),
      );
      api.clientLog("debug", "document_history_update", {
        changedRoots,
      });
      return {
        past: [...current.past.slice(-99), current.present],
        present: deepClone(document),
        future: [],
      };
    });
  }, []);

  const reset = useCallback((document: InputDocument) => {
    setState({ past: [], present: deepClone(document), future: [] });
  }, []);

  const undo = useCallback(() => {
    setState((current) => {
      const previous = current.past.at(-1);
      if (!previous) return current;
      return {
        past: current.past.slice(0, -1),
        present: previous,
        future: [current.present, ...current.future],
      };
    });
  }, []);

  const redo = useCallback(() => {
    setState((current) => {
      const next = current.future[0];
      if (!next) return current;
      return {
        past: [...current.past, current.present],
        present: next,
        future: current.future.slice(1),
      };
    });
  }, []);

  return {
    document: state.present,
    update,
    reset,
    undo,
    redo,
    canUndo: state.past.length > 0,
    canRedo: state.future.length > 0,
  };
};
