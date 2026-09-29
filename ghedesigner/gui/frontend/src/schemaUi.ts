import type { RJSFSchema, UiSchema } from "@rjsf/utils";

import type { JsonObject } from "./types";
import { isJsonObject } from "./types";
import { toTitleCase } from "./titleCase";

const cloneSchemaValue = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(cloneSchemaValue);
  if (!isJsonObject(value)) return value;

  const clone: JsonObject = {};
  for (const [name, child] of Object.entries(value)) {
    clone[name] = (name === "title" && typeof child === "string"
      ? toTitleCase(child)
      : cloneSchemaValue(child)) as JsonObject[string];
  }

  const variants = [clone.oneOf, clone.anyOf].filter(Array.isArray).flat();
  if (variants.length > 0 && isJsonObject(clone.properties)) {
    const variantPropertyNames = new Set<string>();
    for (const option of variants) {
      if (!isJsonObject(option) || !isJsonObject(option.properties)) continue;
      for (const name of Object.keys(option.properties)) variantPropertyNames.add(name);
    }
    if (variantPropertyNames.size > 0) {
      for (const name of variantPropertyNames) delete clone.properties[name];
      // This is a display-only schema. Omitting the parent restriction lets
      // selected variant fields validate without exposing an "add property" UI.
      delete clone.additionalProperties;
    }
  }
  return clone;
};

/** Keep variant-owned fields out of the common form area so RJSF renders them once. */
export const deduplicateVariantProperties = (schema: RJSFSchema): RJSFSchema =>
  cloneSchemaValue(schema) as RJSFSchema;

/** Keep workflow-hidden fields from reappearing inside an anyOf/oneOf option. */
export const restrictVariantProperties = (
  options: RJSFSchema["anyOf"],
  allowedProperties: string[],
): RJSFSchema[] =>
  (options ?? [])
    .filter((option): option is RJSFSchema => {
      if (!isJsonObject(option)) return false;
      return (
        !Array.isArray(option.required) ||
        option.required.every((name) => typeof name === "string" && allowedProperties.includes(name))
      );
    })
    .map((option) => ({
      ...option,
      ...(isJsonObject(option.properties)
        ? {
            properties: Object.fromEntries(
              Object.entries(option.properties).filter(([name]) => allowedProperties.includes(name)),
            ),
          }
        : {}),
    }));

const resolveSchema = (schema: RJSFSchema, definitions: JsonObject | undefined): RJSFSchema => {
  if (typeof schema.$ref !== "string" || !schema.$ref.startsWith("#/$defs/") || !definitions) return schema;
  const definition = definitions[schema.$ref.slice("#/$defs/".length)];
  return isJsonObject(definition) ? (definition as RJSFSchema) : schema;
};

const variantDiscriminatorNames = (schema: RJSFSchema): string[] => {
  const options = schema.oneOf;
  if (!Array.isArray(options) || options.length === 0) return [];
  const first = options[0];
  if (!isJsonObject(first) || !isJsonObject(first.properties)) return [];
  return Object.keys(first.properties).filter((name) =>
    options.every((option) => {
      if (!isJsonObject(option) || !isJsonObject(option.properties)) return false;
      const optionProperty = option.properties[name];
      return isJsonObject(optionProperty) && "const" in optionProperty;
    }),
  );
};

const mergeUiSchema = (target: UiSchema, source: UiSchema): void => {
  for (const [name, value] of Object.entries(source)) {
    const existing = target[name];
    if (isJsonObject(existing) && isJsonObject(value)) {
      mergeUiSchema(existing as UiSchema, value as UiSchema);
    } else {
      target[name] = value;
    }
  }
};

/** Hide enum fields that duplicate RJSF's oneOf variant selector. */
export const variantDiscriminatorUiSchema = (
  sourceSchema: RJSFSchema,
  definitions?: JsonObject,
): UiSchema => {
  const schema = resolveSchema(sourceSchema, definitions);
  if (!isJsonObject(schema.properties)) return {};

  const uiSchema: UiSchema = {};
  if (
    (Array.isArray(schema.oneOf) && schema.oneOf.length > 1) ||
    (Array.isArray(schema.anyOf) && schema.anyOf.length > 1)
  ) {
    uiSchema["ui:classNames"] = "variant-section";
  }
  for (const name of variantDiscriminatorNames(schema)) uiSchema[name] = { "ui:widget": "hidden" };
  for (const [name, propertySchema] of Object.entries(schema.properties)) {
    if (!isJsonObject(propertySchema)) continue;
    const resolvedProperty = resolveSchema(propertySchema as RJSFSchema, definitions);
    const nested = variantDiscriminatorUiSchema(resolvedProperty, definitions);
    if (Object.keys(nested).length > 0) {
      const current = isJsonObject(uiSchema[name]) ? uiSchema[name] : {};
      uiSchema[name] = { ...current, ...nested };
    }
  }
  if (Array.isArray(schema.oneOf)) {
    for (const option of schema.oneOf) {
      if (isJsonObject(option)) {
        mergeUiSchema(uiSchema, variantDiscriminatorUiSchema(option as RJSFSchema, definitions));
      }
    }
  }
  return uiSchema;
};
