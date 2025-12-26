#!/usr/bin/env python3
"""
Run with:
    pip install dash plotly pandas
    python app.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import dash
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, dcc, html, no_update

# ----------------------------------------------------------------------
# Data sources (edit paths as needed)
# ----------------------------------------------------------------------
DATA_FILES = {
    "1-bldg, 1 GHE": "/home/mitchute/Projects/GHEDesigner/ghedesigner/tests/test_data/simulate_1_pipe_1_ghe_1_bldg_district.csv",
    "6-bldg, 3-GHE": "/home/mitchute/Projects/GHEDesigner/ghedesigner/tests/test_data/simulate_1_pipe_3_ghe_6_bldg_district.csv",
}


def load_dataset(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path_str}")
    return pd.read_csv(path)


# ----------------------------------------------------------------------
# Helpers to extract building + GHE + Network info from column names
# ----------------------------------------------------------------------
def get_buildings(df: pd.DataFrame) -> List[str]:
    # column format: "building1:Q_htg [W]"
    buildings = {col.split(":", 1)[0] for col in df.columns if col.startswith("building")}
    return sorted(buildings)


def get_building_metrics(df: pd.DataFrame) -> List[str]:
    metrics = set()
    for col in df.columns:
        if col.startswith("building") and ":" in col:
            _, metric = col.split(":", 1)
            metrics.add(metric)
    return sorted(metrics)


def get_ghe_metrics(df: pd.DataFrame) -> List[str]:
    # column format: "ghe1:EFT [C]"
    metrics = set()
    for col in df.columns:
        if col.startswith("ghe") and ":" in col:
            _, metric = col.split(":", 1)
            metrics.add(metric)
    return sorted(metrics)


def get_network_metrics(df: pd.DataFrame) -> List[str]:
    # column format: "Network:M_flow [kg/s]"
    metrics = set()
    for col in df.columns:
        if col.startswith("Network") and ":" in col:
            _, metric = col.split(":", 1)
            metrics.add(metric)
    return sorted(metrics)


def compute_dataset_meta(df: pd.DataFrame) -> Dict[str, List[str]]:
    return {
        "buildings": get_buildings(df),
        "bldg_metrics": get_building_metrics(df),
        "ghe_metrics": get_ghe_metrics(df),
        "network_metrics": get_network_metrics(df),
    }


# ----------------------------------------------------------------------
# App layout
# ----------------------------------------------------------------------
app = Dash(__name__)
app.title = "District GHE Dashboard"

app.layout = html.Div(
    style={"fontFamily": "system-ui, sans-serif", "margin": "20px"},
    children=[
        # Stores for linked / persistent ranges + runtime-loaded datasets
        dcc.Store(id="x-range-store"),
        dcc.Store(id="building-y-range-store"),
        dcc.Store(id="ghe-y-range-store"),
        dcc.Store(id="network-y-range-store"),
        dcc.Store(id="datasets-store"),  # {dataset_name: [records...]}
        dcc.Store(id="dataset-meta-store"),  # {dataset_name: {buildings, bldg_metrics, ghe_metrics, network_metrics}}
        html.H1("District GHE Dashboard", style={"marginBottom": "0.5rem"}),
        html.P(
            "Interactive dashboard for district simulation CSV outputs.",
            style={"color": "#555", "marginBottom": "1.0rem"},
        ),
        # Controls
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(5, minmax(220px, 260px))",
                "gap": "1rem",
                "marginBottom": "0.75rem",
                "alignItems": "end",
            },
            children=[
                html.Div(
                    children=[
                        html.Label("Dataset", style={"fontWeight": "600"}),
                        dcc.Dropdown(
                            id="dataset-dropdown",
                            options=[{"label": name, "value": name} for name in DATA_FILES],
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
                            options=[],  # populated by callback
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
                            options=[],  # populated by callback
                            value=None,
                            clearable=False,
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Label("Network metric", style={"fontWeight": "600"}),
                        dcc.Dropdown(
                            id="network-metric-dropdown",
                            options=[],  # populated by callback
                            value=None,
                            clearable=False,
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Label("Data", style={"fontWeight": "600"}),
                        html.Button(
                            "Reload CSV files",
                            id="reload-button",
                            n_clicks=0,
                            style={"width": "100%"},
                        ),
                    ]
                ),
            ],
        ),
        html.Div(
            id="reload-status",
            style={"color": "#555", "marginBottom": "1.5rem"},
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
                html.Div(
                    children=[
                        html.H3(
                            "Network time series",
                            style={"marginBottom": "0.5rem"},
                        ),
                        dcc.Graph(
                            id="network-graph",
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
    Output("datasets-store", "data"),
    Output("dataset-meta-store", "data"),
    Output("reload-status", "children"),
    Input("reload-button", "n_clicks"),
    prevent_initial_call=False,
)
def reload_datasets(n_clicks: int):
    """(Re)load CSVs from disk into dcc.Store, and compute per-dataset metadata."""
    datasets_data: Dict[str, List[Dict[str, Any]]] = {}
    meta_data: Dict[str, Dict[str, List[str]]] = {}

    try:
        for name, path in DATA_FILES.items():
            df = load_dataset(path)
            datasets_data[name] = df.to_dict("records")
            meta_data[name] = compute_dataset_meta(df)
    except Exception as e:
        # Keep previous state if reload fails
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = f"Reload failed at {ts}: {e}"
        return no_update, no_update, status

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = f"Reloaded CSV files at {ts}." if (n_clicks and n_clicks > 0) else f"Loaded CSV files at {ts}."

    return datasets_data, meta_data, status


@app.callback(
    Output("bldg-metric-dropdown", "options"),
    Output("bldg-metric-dropdown", "value"),
    Output("ghe-metric-dropdown", "options"),
    Output("ghe-metric-dropdown", "value"),
    Output("network-metric-dropdown", "options"),
    Output("network-metric-dropdown", "value"),
    Input("dataset-dropdown", "value"),
    Input("dataset-meta-store", "data"),
)
def update_dropdowns(dataset_name: str, meta_store: Optional[Dict[str, Any]]):
    if not meta_store or dataset_name not in meta_store:
        return [], None, [], None, [], None

    meta = meta_store[dataset_name]

    bldg_metrics = meta.get("bldg_metrics", []) or []
    ghe_metrics = meta.get("ghe_metrics", []) or []
    network_metrics = meta.get("network_metrics", []) or []

    bldg_metric_options = [{"label": m, "value": m} for m in bldg_metrics]
    ghe_metric_options = [{"label": m, "value": m} for m in ghe_metrics]
    network_metric_options = [{"label": m, "value": m} for m in network_metrics]

    bldg_metric_value = bldg_metrics[0] if bldg_metrics else None
    ghe_metric_value = ghe_metrics[0] if ghe_metrics else None
    network_metric_value = network_metrics[0] if network_metrics else None

    return (
        bldg_metric_options,
        bldg_metric_value,
        ghe_metric_options,
        ghe_metric_value,
        network_metric_options,
        network_metric_value,
    )


@app.callback(
    Output("building-graph", "figure"),
    Input("dataset-dropdown", "value"),
    Input("bldg-metric-dropdown", "value"),
    Input("x-range-store", "data"),
    Input("building-y-range-store", "data"),
    Input("datasets-store", "data"),
)
def update_building_graph(
    dataset_name: str,
    bldg_metric: Optional[str],
    x_range: Optional[List[Any]],
    y_range: Optional[List[Any]],
    datasets_store: Optional[Dict[str, Any]],
):
    if not datasets_store or dataset_name not in datasets_store:
        return px.line(title="No data loaded")

    df = pd.DataFrame(datasets_store[dataset_name])

    if not bldg_metric:
        return px.line(title="No building metric selected")

    if "Hour" not in df.columns:
        return px.line(title="Missing required column: Hour")

    # Find all building columns for the chosen metric
    bldg_cols = [
        col for col in df.columns if col.startswith("building") and ":" in col and col.split(":", 1)[1] == bldg_metric
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
    Input("datasets-store", "data"),
)
def update_ghe_graph(
    dataset_name: str,
    ghe_metric: Optional[str],
    x_range: Optional[List[Any]],
    y_range: Optional[List[Any]],
    datasets_store: Optional[Dict[str, Any]],
):
    if not datasets_store or dataset_name not in datasets_store:
        return px.line(title="No data loaded")

    df = pd.DataFrame(datasets_store[dataset_name])

    if not ghe_metric:
        return px.line(title="No GHE metric selected")

    if "Hour" not in df.columns:
        return px.line(title="Missing required column: Hour")

    # Find all GHE columns for the chosen metric
    ghe_cols = [
        col for col in df.columns if col.startswith("ghe") and ":" in col and col.split(":", 1)[1] == ghe_metric
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


@app.callback(
    Output("network-graph", "figure"),
    Input("dataset-dropdown", "value"),
    Input("network-metric-dropdown", "value"),
    Input("x-range-store", "data"),
    Input("network-y-range-store", "data"),
    Input("datasets-store", "data"),
)
def update_network_graph(
    dataset_name: str,
    network_metric: Optional[str],
    x_range: Optional[List[Any]],
    y_range: Optional[List[Any]],
    datasets_store: Optional[Dict[str, Any]],
):
    if not datasets_store or dataset_name not in datasets_store:
        return px.line(title="No data loaded")

    df = pd.DataFrame(datasets_store[dataset_name])

    if not network_metric:
        return px.line(title="No Network metric selected")

    if "Hour" not in df.columns:
        return px.line(title="Missing required column: Hour")

    col = f"Network:{network_metric}"
    if col not in df.columns:
        # Some files may use a different capitalization or prefix; fall back to search
        candidates = [
            c for c in df.columns if c.startswith("Network") and ":" in c and c.split(":", 1)[1] == network_metric
        ]
        if not candidates:
            return px.line(title=f"No Network column for metric '{network_metric}' in this dataset")
        col = candidates[0]

    fig = px.line(
        df,
        x="Hour",
        y=col,
        title=f"Network – {network_metric} vs Hour",
    )

    if x_range and isinstance(x_range, list) and len(x_range) == 2:
        fig.update_xaxes(range=x_range)
    if y_range and isinstance(y_range, list) and len(y_range) == 2:
        fig.update_yaxes(range=y_range)

    fig.update_layout(
        xaxis_title="Hour",
        yaxis_title=network_metric,
        margin=dict(l=40, r=10, t=40, b=40),
        height=350,
    )
    return fig


# Shared range sync: read relayoutData, store x-range and each panel's y-range
@app.callback(
    Output("x-range-store", "data"),
    Output("building-y-range-store", "data"),
    Output("ghe-y-range-store", "data"),
    Output("network-y-range-store", "data"),
    Input("building-graph", "relayoutData"),
    Input("ghe-graph", "relayoutData"),
    Input("network-graph", "relayoutData"),
    State("x-range-store", "data"),
    State("building-y-range-store", "data"),
    State("ghe-y-range-store", "data"),
    State("network-y-range-store", "data"),
    prevent_initial_call=True,
)
def sync_ranges(
    building_relayout,
    ghe_relayout,
    network_relayout,
    current_x,
    current_building_y,
    current_ghe_y,
    current_network_y,
):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update, no_update

    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    # Start from existing ranges
    new_x = current_x
    new_building_y = current_building_y
    new_ghe_y = current_ghe_y
    new_network_y = current_network_y

    def update_from_relayout(relayout: dict, y_target: str):
        nonlocal new_x, new_building_y, new_ghe_y, new_network_y
        # X-axis changes
        if "xaxis.range[0]" in relayout and "xaxis.range[1]" in relayout:
            new_x = [relayout["xaxis.range[0]"], relayout["xaxis.range[1]"]]
        elif relayout.get("xaxis.autorange", False):
            new_x = None

        # Y-axis changes
        if "yaxis.range[0]" in relayout and "yaxis.range[1]" in relayout:
            y_val = [relayout["yaxis.range[0]"], relayout["yaxis.range[1]"]]
            if y_target == "building":
                new_building_y = y_val
            elif y_target == "ghe":
                new_ghe_y = y_val
            elif y_target == "network":
                new_network_y = y_val
        elif relayout.get("yaxis.autorange", False):
            if y_target == "building":
                new_building_y = None
            elif y_target == "ghe":
                new_ghe_y = None
            elif y_target == "network":
                new_network_y = None

    if trigger == "building-graph":
        update_from_relayout(building_relayout or {}, "building")
    elif trigger == "ghe-graph":
        update_from_relayout(ghe_relayout or {}, "ghe")
    elif trigger == "network-graph":
        update_from_relayout(network_relayout or {}, "network")

    return new_x, new_building_y, new_ghe_y, new_network_y


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
