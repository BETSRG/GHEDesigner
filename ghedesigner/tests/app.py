#!/usr/bin/env python3
"""
Dash dashboard using:
  - output_simple_district.csv
  - output_3_bldg_3_ghe_district.csv

Run with:
    pip install dash plotly pandas
    python app.py
"""

import dash
from dash import Dash, dcc, html, Input, Output, State, no_update
import plotly.express as px
import pandas as pd
from pathlib import Path

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
DATA_FILES = {
    "Simple district (1 bldg, 1 GHE)": "output_simple_district.csv",
    "3-bldg / 3-GHE district": "output_3_bldg_3_ghe_district.csv",
}


def load_dataset(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path_str} in the current directory")
    return pd.read_csv(path)


datasets = {name: load_dataset(path) for name, path in DATA_FILES.items()}


# ----------------------------------------------------------------------
# Helpers to extract building + GHE info from column names
# ----------------------------------------------------------------------
def get_buildings(df: pd.DataFrame):
    # column format: "building1:Q_htg [W]"
    buildings = {col.split(":", 1)[0] for col in df.columns if col.startswith("building")}
    return sorted(buildings)


def get_building_metrics(df: pd.DataFrame):
    metrics = set()
    for col in df.columns:
        if col.startswith("building") and ":" in col:
            _, metric = col.split(":", 1)
            metrics.add(metric)
    return sorted(metrics)


def get_ghe_metrics(df: pd.DataFrame):
    # column format: "ghe1:EFT [C]"
    metrics = set()
    for col in df.columns:
        if col.startswith("ghe") and ":" in col:
            _, metric = col.split(":", 1)
            metrics.add(metric)
    return sorted(metrics)


# Precompute options for each dataset
dataset_meta = {}
for name, df in datasets.items():
    dataset_meta[name] = {
        "buildings": get_buildings(df),
        "bldg_metrics": get_building_metrics(df),
        "ghe_metrics": get_ghe_metrics(df),
    }

# ----------------------------------------------------------------------
# App layout
# ----------------------------------------------------------------------
app = Dash(__name__)
app.title = "District GHE Dashboard"

app.layout = html.Div(
    style={"fontFamily": "system-ui, sans-serif", "margin": "20px"},
    children=[
        # Stores for linked / persistent ranges
        dcc.Store(id="x-range-store"),
        dcc.Store(id="building-y-range-store"),
        dcc.Store(id="ghe-y-range-store"),

        html.H1("District GHE Dashboard", style={"marginBottom": "0.5rem"}),

        html.P(
            "Interactive example using output_simple_district.csv and "
            "output_3_bldg_3_ghe_district.csv",
            style={"color": "#555", "marginBottom": "1.5rem"},
        ),

        # Controls
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(3, minmax(220px, 260px))",
                "gap": "1rem",
                "marginBottom": "1.5rem",
            },
            children=[
                html.Div(
                    children=[
                        html.Label("Dataset", style={"fontWeight": "600"}),
                        dcc.Dropdown(
                            id="dataset-dropdown",
                            options=[
                                {"label": name, "value": name}
                                for name in DATA_FILES.keys()
                            ],
                            value=list(DATA_FILES.keys())[0],
                            clearable=False,
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Label("Building metric", style={"fontWeight": "600"}),
                        dcc.Dropdown(
                            id="bldg-metric-dropdown",
                            options=[],   # populated by callback
                            value=None,
                            clearable=False,
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Label("GHE metric", style={"fontWeight": "600"}),
                        dcc.Dropdown(
                            id="ghe-metric-dropdown",
                            options=[],   # populated by callback
                            value=None,
                            clearable=False,
                        ),
                    ]
                ),
            ],
        ),

        # Plots stacked vertically
        html.Div(
            style={
                "display": "flex",
                "flexDirection": "column",
                "gap": "1.5rem",
            },
            children=[
                html.Div(
                    children=[
                        html.H3(
                            "Building time series (all buildings)",
                            style={"marginBottom": "0.5rem"},
                        ),
                        dcc.Graph(
                            id="building-graph",
                            style={"height": "400px"},
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.H3(
                            "GHE time series (all GHEs)",
                            style={"marginBottom": "0.5rem"},
                        ),
                        dcc.Graph(
                            id="ghe-graph",
                            style={"height": "400px"},
                        ),
                    ]
                ),
            ],
        ),
    ],
)

# ----------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------


@app.callback(
    Output("bldg-metric-dropdown", "options"),
    Output("bldg-metric-dropdown", "value"),
    Output("ghe-metric-dropdown", "options"),
    Output("ghe-metric-dropdown", "value"),
    Input("dataset-dropdown", "value"),
)
def update_dropdowns(dataset_name):
    meta = dataset_meta[dataset_name]

    bldg_metric_options = [
        {"label": m, "value": m} for m in meta["bldg_metrics"]
    ]
    bldg_metric_value = meta["bldg_metrics"][0] if meta["bldg_metrics"] else None

    ghe_metric_options = [
        {"label": m, "value": m} for m in meta["ghe_metrics"]
    ]
    ghe_metric_value = meta["ghe_metrics"][0] if meta["ghe_metrics"] else None

    return (
        bldg_metric_options,
        bldg_metric_value,
        ghe_metric_options,
        ghe_metric_value,
    )


@app.callback(
    Output("building-graph", "figure"),
    Input("dataset-dropdown", "value"),
    Input("bldg-metric-dropdown", "value"),
    Input("x-range-store", "data"),
    Input("building-y-range-store", "data"),
)
def update_building_graph(dataset_name, bldg_metric, x_range, y_range):
    df = datasets[dataset_name].copy()

    if not bldg_metric:
        return px.line(title="No building metric selected")

    # Find all building columns for the chosen metric
    bldg_cols = [
        col for col in df.columns
        if col.startswith("building")
        and ":" in col
        and col.split(":", 1)[1] == bldg_metric
    ]

    if not bldg_cols:
        return px.line(title=f"No building columns for metric '{bldg_metric}' in this dataset")

    # Melt into long format for multi-line plot
    melted = df.melt(
        id_vars=["Hour"],
        value_vars=bldg_cols,
        var_name="Building",
        value_name="value",
    )

    # "building1:Q_htg [W]" -> "building1"
    melted["Building"] = melted["Building"].str.split(":", n=1).str[0]

    fig = px.line(
        melted,
        x="Hour",
        y="value",
        color="Building",
        title=f"Buildings – {bldg_metric} vs Hour",
    )

    if x_range and isinstance(x_range, list) and len(x_range) == 2:
        fig.update_xaxes(range=x_range)
    if y_range and isinstance(y_range, list) and len(y_range) == 2:
        fig.update_yaxes(range=y_range)

    fig.update_layout(
        xaxis_title="Hour",
        yaxis_title=bldg_metric,
        margin=dict(l=40, r=10, t=40, b=40),
        legend_title="Building",
        height=350,
    )
    return fig


@app.callback(
    Output("ghe-graph", "figure"),
    Input("dataset-dropdown", "value"),
    Input("ghe-metric-dropdown", "value"),
    Input("x-range-store", "data"),
    Input("ghe-y-range-store", "data"),
)
def update_ghe_graph(dataset_name, ghe_metric, x_range, y_range):
    df = datasets[dataset_name].copy()

    if not ghe_metric:
        return px.line(title="No GHE metric selected")

    # Find all GHE columns for the chosen metric
    ghe_cols = [
        col for col in df.columns
        if col.startswith("ghe")
        and ":" in col
        and col.split(":", 1)[1] == ghe_metric
    ]

    if not ghe_cols:
        return px.line(title=f"No GHE columns for metric '{ghe_metric}' in this dataset")

    # Melt into long format for multi-line plot
    melted = df.melt(
        id_vars=["Hour"],
        value_vars=ghe_cols,
        var_name="GHE",
        value_name="value",
    )

    # "ghe1:EFT [C]" -> "ghe1"
    melted["GHE"] = melted["GHE"].str.split(":", n=1).str[0]

    fig = px.line(
        melted,
        x="Hour",
        y="value",
        color="GHE",
        title=f"GHEs – {ghe_metric} vs Hour",
    )

    if x_range and isinstance(x_range, list) and len(x_range) == 2:
        fig.update_xaxes(range=x_range)
    if y_range and isinstance(y_range, list) and len(y_range) == 2:
        fig.update_yaxes(range=y_range)

    fig.update_layout(
        xaxis_title="Hour",
        yaxis_title=ghe_metric,
        margin=dict(l=40, r=10, t=40, b=40),
        legend_title="GHE",
        height=350,
    )
    return fig


# Shared range sync: read relayoutData, store x-range and each panel's y-range
@app.callback(
    Output("x-range-store", "data"),
    Output("building-y-range-store", "data"),
    Output("ghe-y-range-store", "data"),
    Input("building-graph", "relayoutData"),
    Input("ghe-graph", "relayoutData"),
    State("x-range-store", "data"),
    State("building-y-range-store", "data"),
    State("ghe-y-range-store", "data"),
    prevent_initial_call=True,
)
def sync_ranges(building_relayout, ghe_relayout,
                current_x, current_building_y, current_ghe_y):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update

    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    # Start from existing ranges
    new_x = current_x
    new_building_y = current_building_y
    new_ghe_y = current_ghe_y

    if trigger == "building-graph":
        relayout = building_relayout or {}
        # X-axis changes
        if "xaxis.range[0]" in relayout and "xaxis.range[1]" in relayout:
            new_x = [relayout["xaxis.range[0]"], relayout["xaxis.range[1]"]]
        elif relayout.get("xaxis.autorange", False):
            new_x = None

        # Y-axis changes for building panel
        if "yaxis.range[0]" in relayout and "yaxis.range[1]" in relayout:
            new_building_y = [relayout["yaxis.range[0]"], relayout["yaxis.range[1]"]]
        elif relayout.get("yaxis.autorange", False):
            new_building_y = None

    elif trigger == "ghe-graph":
        relayout = ghe_relayout or {}
        # X-axis changes
        if "xaxis.range[0]" in relayout and "xaxis.range[1]" in relayout:
            new_x = [relayout["xaxis.range[0]"], relayout["xaxis.range[1]"]]
        elif relayout.get("xaxis.autorange", False):
            new_x = None

        # Y-axis changes for GHE panel
        if "yaxis.range[0]" in relayout and "yaxis.range[1]" in relayout:
            new_ghe_y = [relayout["yaxis.range[0]"], relayout["yaxis.range[1]"]]
        elif relayout.get("yaxis.autorange", False):
            new_ghe_y = None

    return new_x, new_building_y, new_ghe_y


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
