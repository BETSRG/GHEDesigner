from pathlib import Path
import pandas as pd

csv_path  = Path(r"C:\Users\Student\Desktop\Loading Profiles\csv\WA_SEATTLE\4C_USA_WA_SEATTLE_LargeOffice.csv")
json_path = Path(r"C:\Users\Student\Desktop\Loading Profiles\json\WA_SEATTLE\WA_SEATTLE_LargeOffice.json")

json_path.parent.mkdir(parents=True, exist_ok=True)

df    = pd.read_csv(csv_path)
loads = df["TotalSensibleLoad"].tolist()

with json_path.open("w", encoding="utf-8") as f:
    f.write("{\n")
    f.write('  "loads": [\n')
    total = len(loads)
    for idx, val in enumerate(loads):
        if idx % 10 == 0:
            f.write("    ")
        f.write(str(val))
        if idx < total - 1:
            f.write(", ")
        if idx % 10 == 9 or idx == total - 1:
            f.write("\n")
    f.write("  ]\n")
    f.write("}\n")

print(f"Wrote formatted JSON to {json_path}")