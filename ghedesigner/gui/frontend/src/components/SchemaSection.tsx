import Form from "@rjsf/core";
import { customizeValidator } from "@rjsf/validator-ajv8";
import type { RJSFSchema } from "@rjsf/utils";
import { useEffect, useState } from "react";

import { api } from "../api";
import { deduplicateVariantProperties } from "../schemaUi";
import type { InputDocument, JsonObject } from "../types";
import { deepClone, isJsonObject } from "../types";
import { FilePathWidget } from "./FilePathWidget";

const unitFormats = [
  "-",
  "Centigrade",
  "Degrees",
  "Degrees Celsius",
  "Joules/Meter^3-Kelvin",
  "Kilograms/Second",
  "Liters/Second",
  "Meters",
  "Pascals",
  "Percent",
  "Square Meters",
  "Watts",
  "Watts/Celsius",
  "Watts/Celsius^2",
  "Watts/Meter-Kelvin",
  "Watts/Watts",
  "Watts/Watts-Celsius",
  "Watts/Watts-Celsius^2",
  "Years",
  "file-path",
];
export const schemaValidator = customizeValidator({
  customFormats: Object.fromEntries(unitFormats.map((format) => [format, () => true])),
});

interface SchemaSectionProps {
  document: InputDocument;
  rootSchema: JsonObject;
  property: string;
  title: string;
  description: string;
  allowedProperties?: string[];
  requiredProperties?: string[];
  enumExclusions?: Record<string, string[]>;
  onChange: (document: InputDocument) => void;
}

export function SchemaSection({
  document,
  rootSchema,
  property,
  title,
  description,
  allowedProperties,
  requiredProperties,
  enumExclusions,
  onChange,
}: SchemaSectionProps) {
  const sourceData = isJsonObject(document[property]) ? document[property] : {};
  const [formData, setFormData] = useState(() => deepClone(sourceData));
  useEffect(() => setFormData(deepClone(sourceData)), [sourceData]);
  const properties = rootSchema.properties;
  const definitions = rootSchema.$defs;
  if (!isJsonObject(properties) || !isJsonObject(properties[property])) {
    return <div className="empty-state">Schema information for this section is unavailable.</div>;
  }

  const propertySchema = properties[property] as RJSFSchema;
  const referencedDefinition =
    typeof propertySchema.$ref === "string" && propertySchema.$ref.startsWith("#/$defs/") && isJsonObject(definitions)
      ? definitions[propertySchema.$ref.slice("#/$defs/".length)]
      : undefined;
  const sourceSchema: RJSFSchema = isJsonObject(referencedDefinition)
    ? { ...(referencedDefinition as RJSFSchema), ...propertySchema }
    : propertySchema;
  if (isJsonObject(referencedDefinition)) delete sourceSchema.$ref;
  const visibleProperties = isJsonObject(sourceSchema.properties)
    ? Object.fromEntries(
        Object.entries(sourceSchema.properties)
          .filter(([key]) => !allowedProperties || allowedProperties.includes(key))
          .map(([key, value]) => {
            const excluded = enumExclusions?.[key];
            if (!excluded || !isJsonObject(value) || !Array.isArray(value.enum)) return [key, value];
            return [
              key,
              {
                ...(value as RJSFSchema),
                enum: value.enum.filter((item) => !excluded.includes(String(item))),
              },
            ];
          }),
      )
    : sourceSchema.properties;
  const sectionSchema: RJSFSchema = {
    ...sourceSchema,
    ...(allowedProperties
      ? {
          properties: visibleProperties,
          required: Array.from(new Set([...(sourceSchema.required ?? []), ...(requiredProperties ?? [])])).filter(
            (key) => allowedProperties.includes(key),
          ),
        }
      : requiredProperties
        ? { required: Array.from(new Set([...(sourceSchema.required ?? []), ...requiredProperties])) }
        : {}),
    ...(isJsonObject(definitions) ? { $defs: definitions as RJSFSchema } : {}),
  };
  if (allowedProperties) {
    delete sectionSchema.oneOf;
    delete sectionSchema.anyOf;
  }
  const displaySchema = deduplicateVariantProperties(sectionSchema);

  return (
    <section className="editor-page">
      <div className="page-heading">
        <div>
          <span className="eyebrow">Input Section</span>
          <h1>{title}</h1>
          <p>{description}</p>
        </div>
      </div>
      <div className="form-surface schema-form">
        <Form
          schema={displaySchema}
          formData={formData}
          validator={schemaValidator}
          widgets={{ "file-path": FilePathWidget }}
          liveValidate={false}
          noHtml5Validate
          showErrorList={false}
          experimental_defaultFormStateBehavior={{
            arrayMinItems: { populate: "never" },
            emptyObjectFields: "skipDefaults",
            allOf: "skipDefaults",
            constAsDefaults: "never",
          }}
          uiSchema={{
            ...(allowedProperties ? { "ui:order": allowedProperties } : {}),
            "ui:submitButtonOptions": { norender: true },
          }}
          onChange={(event) => setFormData(event.formData)}
        >
          <></>
        </Form>
        <div className="schema-form-actions">
          <span>Changes in this section are staged until applied.</span>
          <button
            type="button"
            className="button primary"
            onClick={() => {
              onChange({ ...document, [property]: deepClone(formData) });
              api.clientLog("info", "schema_section_applied", { section: property });
            }}
          >
            Apply section
          </button>
        </div>
      </div>
    </section>
  );
}
