import Form from "@rjsf/core";
import type { RJSFSchema } from "@rjsf/utils";
import { FilePlus2, Pencil, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { api } from "../api";
import {
  deduplicateVariantProperties,
  restrictVariantProperties,
  variantDiscriminatorUiSchema,
} from "../schemaUi";
import type { InputDocument, JsonObject, WorkflowMode } from "../types";
import { deepClone, isJsonObject } from "../types";
import { NoGoBoundariesField, PropertyBoundaryField } from "./BoundaryFields";
import { CompactNumberArrayField } from "./CompactNumberArrayField";
import { FilePathWidget } from "./FilePathWidget";
import { ManualBorefieldField } from "./ManualBorefieldField";
import { schemaValidator } from "./SchemaSection";

interface CollectionSectionProps {
  document: InputDocument;
  property: string;
  title: string;
  description: string;
  rootSchema: JsonObject;
  allowedProperties?: string[];
  workflow?: WorkflowMode;
  onChange: (document: InputDocument) => void;
}

export function CollectionSection({
  document,
  property,
  title,
  description,
  rootSchema,
  allowedProperties,
  workflow,
  onChange,
}: CollectionSectionProps) {
  const collection = isJsonObject(document[property]) ? document[property] : {};
  const ids = useMemo(() => Object.keys(collection).sort(), [collection]);
  const [selected, setSelected] = useState<string | null>(() => ids[0] ?? null);
  const selectedValue = selected && isJsonObject(collection[selected]) ? collection[selected] : null;
  const [formData, setFormData] = useState<JsonObject>(() => deepClone(selectedValue ?? {}));
  const manualPreDesigned =
    property === "ground_heat_exchanger" &&
    isJsonObject(formData.pre_designed) &&
    formData.pre_designed.arrangement === "MANUAL";

  useEffect(() => {
    if (selected && selected in collection) return;
    setSelected(ids[0] ?? null);
  }, [collection, ids, selected]);

  useEffect(() => {
    setFormData(deepClone(selectedValue ?? {}));
  }, [selected, selectedValue]);

  const itemSchema = useMemo<RJSFSchema | null>(() => {
    const properties = rootSchema.properties;
    if (!isJsonObject(properties) || !isJsonObject(properties[property])) return null;
    const collectionSchema = properties[property];
    if (!isJsonObject(collectionSchema.additionalProperties)) return null;
    const definitions = rootSchema.$defs;
    const sourceSchema = collectionSchema.additionalProperties as RJSFSchema;
    const visibleProperties =
      allowedProperties && isJsonObject(sourceSchema.properties)
        ? Object.fromEntries(
            Object.entries(sourceSchema.properties).filter(([key]) => allowedProperties.includes(key)),
          )
        : sourceSchema.properties;
    const workflowAnyOf =
      property === "building" && workflow === "building_design"
        ? (sourceSchema.anyOf ?? [])
            .filter((option) => {
              if (!isJsonObject(option) || !Array.isArray(option.required)) return false;
              const required = new Set(option.required);
              return required.has("total_load") || (required.has("heating_load") && required.has("cooling_load"));
            })
            .map((option) => {
              if (!isJsonObject(option) || !Array.isArray(option.required)) return option;
              return {
                ...option,
                title: option.required.includes("total_load")
                  ? "Combined Building Load"
                  : "Separate Heating and Cooling Loads",
              };
            })
        : sourceSchema.anyOf;
    const relevantAnyOf = allowedProperties
      ? restrictVariantProperties(workflowAnyOf, allowedProperties)
      : workflowAnyOf;
    const selectedAnyOf = relevantAnyOf?.length === 1 && isJsonObject(relevantAnyOf[0]) ? relevantAnyOf[0] : undefined;
    const selectedAnyOfRequired =
      selectedAnyOf && Array.isArray(selectedAnyOf.required)
        ? selectedAnyOf.required.filter((key): key is string => typeof key === "string")
        : [];
    const required = Array.from(new Set([...(sourceSchema.required ?? []), ...selectedAnyOfRequired])).filter(
      (key) => !allowedProperties || allowedProperties.includes(key),
    );
    const filteredSchema: RJSFSchema = {
      ...sourceSchema,
      ...(allowedProperties ? { properties: visibleProperties } : {}),
      ...(required.length ? { required } : {}),
      ...(isJsonObject(definitions) ? { $defs: definitions as RJSFSchema } : {}),
    };
    if (allowedProperties) {
      if (relevantAnyOf && relevantAnyOf.length > 1) filteredSchema.anyOf = relevantAnyOf;
      else delete filteredSchema.anyOf;
    }
    const displaySchema = deduplicateVariantProperties(filteredSchema);
    if (property === "ground_heat_exchanger") {
      const displayDefinitions = displaySchema.$defs as Record<string, RJSFSchema> | undefined;
      const loadInputs = displayDefinitions?.load_inputs;
      if (loadInputs && Array.isArray(loadInputs.oneOf)) {
        loadInputs.oneOf = loadInputs.oneOf.filter(
          (option) =>
            isJsonObject(option) &&
            isJsonObject(option.properties) &&
            !("heat_pump_name" in option.properties) &&
            !("heat_pump_cop" in option.properties),
        );
      }
      const displayProperties = displaySchema.properties as Record<string, RJSFSchema> | undefined;
      const preDesigned = displayProperties?.pre_designed;
      if (manualPreDesigned && preDesigned && Array.isArray(preDesigned.oneOf)) {
        const manualOption = preDesigned.oneOf.find(
          (option) =>
            isJsonObject(option) &&
            isJsonObject(option.properties) &&
            isJsonObject(option.properties.arrangement) &&
            option.properties.arrangement.const === "MANUAL",
        );
        if (isJsonObject(manualOption)) {
          const commonProperties = isJsonObject(preDesigned.properties)
            ? (preDesigned.properties as Record<string, RJSFSchema>)
            : {};
          const manualProperties = isJsonObject(manualOption.properties)
            ? (manualOption.properties as Record<string, RJSFSchema>)
            : {};
          const collapsed: RJSFSchema = {
            ...preDesigned,
            ...manualOption,
            properties: { ...commonProperties, ...manualProperties },
          };
          delete collapsed.oneOf;
          displayProperties.pre_designed = collapsed;
        }
      }
    }
    return displaySchema;
  }, [allowedProperties, manualPreDesigned, property, rootSchema, workflow]);

  const itemUiSchema = useMemo(() => {
    const numericSeries = { "ui:field": "compactNumberArray" };
    const variantSelectors = itemSchema
      ? variantDiscriminatorUiSchema(itemSchema, isJsonObject(rootSchema.$defs) ? rootSchema.$defs : undefined)
      : {};
    const variantField = (name: string) => (isJsonObject(variantSelectors[name]) ? variantSelectors[name] : {});
    if (property === "building") {
      return {
        ...variantSelectors,
        "ui:classNames": "variant-section building-load-variant",
        cooling_load: { ...variantField("cooling_load"), load_values: numericSeries },
        heating_load: { ...variantField("heating_load"), load_values: numericSeries },
        total_load: { ...variantField("total_load"), load_values: numericSeries },
        "ui:submitButtonOptions": { norender: true },
      };
    }
    if (property === "ground_heat_exchanger") {
      return {
        ...variantSelectors,
        loads: { ...variantField("loads"), load_values: numericSeries },
        geometric_constraints: {
          ...variantField("geometric_constraints"),
          property_boundary: { "ui:field": "propertyBoundary" },
          no_go_boundaries: { "ui:field": "noGoBoundaries" },
        },
        pre_designed: manualPreDesigned
          ? { "ui:field": "manualBorefield" }
          : variantField("pre_designed"),
        "ui:submitButtonOptions": { norender: true },
      };
    }
    return { ...variantSelectors, "ui:submitButtonOptions": { norender: true } };
  }, [itemSchema, manualPreDesigned, property, rootSchema.$defs]);

  const updateCollection = (nextCollection: JsonObject) => {
    onChange({ ...document, [property]: nextCollection });
  };

  const addItem = () => {
    const id = window.prompt("New globally unique component ID")?.trim();
    if (!id) return;
    if (id in collection) {
      window.alert(`Component '${id}' already exists in this section.`);
      return;
    }
    updateCollection({ ...collection, [id]: {} });
    setSelected(id);
    api.clientLog("info", "component_added", { section: property, id });
  };

  const renameItem = () => {
    if (!selected) return;
    const id = window.prompt("New globally unique component ID", selected)?.trim();
    if (!id || id === selected) return;
    if (id in collection) {
      window.alert(`Component '${id}' already exists in this section.`);
      return;
    }
    const next = { ...collection, [id]: collection[selected] };
    delete next[selected];
    updateCollection(next);
    setSelected(id);
    api.clientLog("info", "component_renamed", { section: property, from: selected, to: id });
  };

  const deleteItem = () => {
    if (!selected || !window.confirm(`Delete '${selected}'? References to it are not removed automatically.`)) return;
    const next = { ...collection };
    delete next[selected];
    updateCollection(next);
    api.clientLog("info", "component_deleted", { section: property, id: selected });
    setSelected(null);
  };

  return (
    <section className="editor-page">
      <div className="page-heading">
        <div>
          <span className="eyebrow">Component Collection</span>
          <h1>{title}</h1>
          <p>{description}</p>
        </div>
        <button type="button" className="button secondary" onClick={addItem}>
          <FilePlus2 size={16} /> Add component
        </button>
      </div>
      <div className="collection-editor">
        <aside className="collection-list form-surface">
          <div className="collection-list-heading">
            <strong>{ids.length} components</strong>
            <small>Select one to edit</small>
          </div>
          {ids.map((id) => (
            <button
              type="button"
              key={id}
              className={selected === id ? "active" : ""}
              onClick={() => setSelected(id)}
            >
              {id}
            </button>
          ))}
          {!ids.length && <p className="empty-inline">No components are defined.</p>}
        </aside>
        <div className="collection-detail form-surface">
          {selected && selectedValue ? (
            <>
              <div className="panel-heading compact-heading">
                <div>
                  <span className="eyebrow">Selected Component</span>
                  <h2>{selected}</h2>
                </div>
                <div className="inline-actions">
                  <button type="button" className="button ghost compact" onClick={renameItem}>
                    <Pencil size={14} /> Rename
                  </button>
                  <button type="button" className="button danger compact" onClick={deleteItem}>
                    <Trash2 size={14} /> Delete
                  </button>
                </div>
              </div>
              <p className="collection-help">Every schema entry is available as a labeled field. Changes are staged until applied.</p>
              {itemSchema ? (
              <div className="schema-form component-schema-form">
                {property === "building" && (
                  <label className="building-load-mode-label" htmlFor="root__XxxOf">
                    Load Input Format
                  </label>
                )}
                <Form
                    key={selected}
                    schema={itemSchema}
                    formData={formData}
                    validator={schemaValidator}
                    fields={{
                      compactNumberArray: CompactNumberArrayField,
                      manualBorefield: ManualBorefieldField,
                      noGoBoundaries: NoGoBoundariesField,
                      propertyBoundary: PropertyBoundaryField,
                    }}
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
                    uiSchema={itemUiSchema}
                    onChange={(event) => setFormData(event.formData as JsonObject)}
                  >
                    <></>
                  </Form>
                  <div className="schema-form-actions">
                    <span>Apply all fields for {selected} together.</span>
                    <button
                      type="button"
                      className="button primary"
                      onClick={() => {
                        const merged = deepClone(selectedValue);
                        if (allowedProperties) {
                          for (const key of allowedProperties) delete merged[key];
                        } else {
                          for (const key of Object.keys(merged)) delete merged[key];
                        }
                        Object.assign(merged, deepClone(formData));
                        updateCollection({ ...collection, [selected]: merged });
                        api.clientLog("info", "component_fields_applied", { section: property, id: selected });
                      }}
                    >
                      Apply component
                    </button>
                  </div>
                </div>
              ) : (
                <p className="empty-inline">Schema fields for this component are unavailable.</p>
              )}
            </>
          ) : (
            <div className="inspector-empty">
              <FilePlus2 size={34} />
              <h2>Add or Select a Component</h2>
              <p>Select one component to edit its complete set of fields.</p>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
