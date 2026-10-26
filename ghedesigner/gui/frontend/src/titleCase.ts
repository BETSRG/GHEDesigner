const LOWERCASE_WORDS = new Set([
  "a",
  "an",
  "and",
  "as",
  "at",
  "by",
  "for",
  "from",
  "in",
  "of",
  "on",
  "or",
  "per",
  "the",
  "to",
  "via",
  "with",
]);

const PRESERVED_WORDS = new Map([
  ["cop", "COP"],
  ["csv", "CSV"],
  ["eft", "EFT"],
  ["ghe", "GHE"],
  ["ghes", "GHEs"],
  ["ghedesigner", "GHEDesigner"],
  ["hp", "HP"],
  ["id", "ID"],
  ["json", "JSON"],
]);

const titleCasePart = (part: string, isBoundary: boolean): string => {
  const lower = part.toLowerCase();
  const preserved = PRESERVED_WORDS.get(lower);
  if (preserved) return preserved;
  if (lower === "x" || lower === "y") return lower.toUpperCase();
  if (!isBoundary && LOWERCASE_WORDS.has(lower)) return lower;
  return lower ? `${lower[0].toUpperCase()}${lower.slice(1)}` : lower;
};

/** Convert a user-facing field or section label to headline-style title case. */
export const toTitleCase = (label: string): string => {
  const words = label.split(/(\s+)/);
  const wordIndexes = words.flatMap((word, index) => (/\S/.test(word) ? [index] : []));
  const firstWord = wordIndexes[0];
  const lastWord = wordIndexes.at(-1);
  return words
    .map((word, index) => {
      if (!/\S/.test(word) || word === "/" || word === "+" || word === "&") return word;
      const parts = word.split("-");
      return parts
        .map((part, partIndex) =>
          titleCasePart(
            part,
            (index === firstWord && partIndex === 0) || (index === lastWord && partIndex === parts.length - 1),
          ),
        )
        .join("-");
    })
    .join("");
};
