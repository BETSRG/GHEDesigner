import { describe, expect, it } from "vitest";

import {
  deduplicateVariantProperties,
  restrictVariantProperties,
  variantDiscriminatorUiSchema,
} from "./schemaUi";

describe("schema-driven form controls", () => {
  it("renders oneOf-owned fields only inside the selected variant", () => {
    const displaySchema = deduplicateVariantProperties({
      type: "object",
      properties: {
        common: { type: "number" },
        load_values: { type: "array", items: { type: "number" } },
        file_path: { type: "string" },
      },
      oneOf: [
        { properties: { load_values: { type: "array", items: { type: "number" } } } },
        { properties: { file_path: { type: "string" } } },
      ],
      additionalProperties: false,
    });

    expect(displaySchema.properties).toEqual({ common: { type: "number" } });
    expect(displaySchema.oneOf?.[0]).toHaveProperty("properties.load_values");
    expect(displaySchema).not.toHaveProperty("additionalProperties");
  });

  it("renders anyOf-owned fields only inside the selected variant", () => {
    const displaySchema = deduplicateVariantProperties({
      type: "object",
      properties: {
        common: { type: "number" },
        heating_load: { type: "object" },
        cooling_load: { type: "object" },
        total_load: { type: "object" },
      },
      anyOf: [
        { properties: { heating_load: { type: "object" }, cooling_load: { type: "object" } } },
        { properties: { total_load: { type: "object" } } },
      ],
      additionalProperties: false,
    });

    expect(displaySchema.properties).toEqual({ common: { type: "number" } });
    expect(displaySchema.anyOf).toHaveLength(2);
    expect(displaySchema).not.toHaveProperty("additionalProperties");
  });

  it("removes workflow-hidden properties from each schema variant", () => {
    expect(
      restrictVariantProperties(
        [
          {
            properties: {
              total_load: { type: "object" },
              max_eft: { type: "number" },
              min_eft: { type: "number" },
            },
            required: ["total_load"],
          },
          { properties: { max_eft: { type: "number" } }, required: ["max_eft"] },
        ],
        ["total_load"],
      ),
    ).toEqual([
      {
        properties: { total_load: { type: "object" } },
        required: ["total_load"],
      },
    ]);
  });

  it("hides a discriminator enum when oneOf already provides its selector", () => {
    expect(
      variantDiscriminatorUiSchema({
        type: "object",
        properties: {
          geometric_constraints: {
            type: "object",
            properties: { method: { type: "string", enum: ["RECTANGLE", "ROWWISE"] } },
            oneOf: [
              { properties: { method: { const: "RECTANGLE" } } },
              { properties: { method: { const: "ROWWISE" } } },
            ],
          },
        },
      }),
    ).toMatchObject({ geometric_constraints: { method: { "ui:widget": "hidden" } } });
    expect(
      variantDiscriminatorUiSchema({
        type: "object",
        properties: { method: { enum: ["A", "B"] } },
        oneOf: [{ properties: { method: { const: "A" } } }, { properties: { method: { const: "B" } } }],
      }),
    ).toHaveProperty("ui:classNames", "variant-section");
  });

  it("does not hide ordinary enum inputs", () => {
    expect(
      variantDiscriminatorUiSchema({
        type: "object",
        properties: { load_method: { type: "string", enum: ["HYBRID", "HOURLY"] } },
      }),
    ).toEqual({});
  });

  it("finds duplicate selectors nested inside a oneOf option", () => {
    expect(
      variantDiscriminatorUiSchema({
        type: "object",
        properties: { method: { type: "string", enum: ["CONSTRAINED"] } },
        oneOf: [
          {
            properties: {
              method: { const: "CONSTRAINED" },
              removal: {
                type: "object",
                properties: { strategy: { type: "string", enum: ["RADIAL", "RIGHT_TOP"] } },
                oneOf: [
                  { properties: { strategy: { const: "RADIAL" } } },
                  { properties: { strategy: { const: "RIGHT_TOP" } } },
                ],
              },
            },
          },
        ],
      }),
    ).toMatchObject({
      method: { "ui:widget": "hidden" },
      removal: { strategy: { "ui:widget": "hidden" } },
    });
  });
});
