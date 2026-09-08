import type { WidgetProps } from "@rjsf/utils";
import { getTemplate } from "@rjsf/utils";

import { PathBrowserButton } from "./PathBrowser";

export function FilePathWidget(props: WidgetProps) {
  const { disabled, onChange, options, readonly, registry, value } = props;
  const BaseInputTemplate = getTemplate("BaseInputTemplate", registry, options);

  return (
    <div className="path-picker">
      <BaseInputTemplate {...props} />
      <PathBrowserButton
        kind="file"
        currentPath={typeof value === "string" ? value : ""}
        disabled={disabled || readonly}
        onSelect={onChange}
      />
    </div>
  );
}
