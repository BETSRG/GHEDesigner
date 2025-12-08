#!/usr/bin/env python3

import json
from pathlib import Path

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------
# Config file handling for persistent CSV paths
# ----------------------------------------------------------------------

CONFIG_FILE = Path("test_outputs/csv_paths.json")


def load_config():
    if CONFIG_FILE.is_file():
        try:
            with CONFIG_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_config(path1: str | None, path2: str | None):
    cfg = {}
    if path1:
        cfg["file1"] = path1
    if path2:
        cfg["file2"] = path2
    try:
        with CONFIG_FILE.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"Failed to save config: {e}")


config = load_config()
initial_path1 = config.get("file1", "")
initial_path2 = config.get("file2", "")

# ----------------------------------------------------------------------
# Dash app
# ----------------------------------------------------------------------

app = dash.Dash(__name__)

app.layout = html.Div(
    style={"margin": "20px"},
    children=[
        html.H2("CSV Series Comparison Tool"),
        html.Div(
            style={"display": "flex", "gap": "20px"},
            children=[
                # File 1 column
                html.Div(
                    style={"flex": "1", "border": "1px solid #ccc", "padding": "10px"},
                    children=[
                        html.H4("File 1"),
                        html.Label("Path to CSV file 1:"),
                        dcc.Input(
                            id="file1-path",
                            type="text",
                            value=initial_path1,
                            placeholder="/path/to/file1.csv",
                            style={"width": "100%", "marginBottom": "8px"},
                        ),
                        html.Div(
                            id="file1-info",
                            style={"fontSize": "0.9em", "marginBottom": "8px"},
                        ),
                        html.Label("Series from File 1:"),
                        dcc.Dropdown(
                            id="column-dropdown-1",
                            placeholder="Select a numeric column",
                        ),
                    ],
                ),
                # File 2 column
                html.Div(
                    style={"flex": "1", "border": "1px solid #ccc", "padding": "10px"},
                    children=[
                        html.H4("File 2"),
                        html.Label("Path to CSV file 2:"),
                        dcc.Input(
                            id="file2-path",
                            type="text",
                            value=initial_path2,
                            placeholder="/path/to/file2.csv",
                            style={"width": "100%", "marginBottom": "8px"},
                        ),
                        html.Div(
                            id="file2-info",
                            style={"fontSize": "0.9em", "marginBottom": "8px"},
                        ),
                        html.Label("Series from File 2:"),
                        dcc.Dropdown(
                            id="column-dropdown-2",
                            placeholder="Select a numeric column",
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            style={"marginTop": "10px", "marginBottom": "10px"},
            children=[
                html.Button(
                    "Load / Reload CSVs",
                    id="reload-button",
                    n_clicks=0,
                    style={"padding": "6px 14px"},
                ),
                html.Span(
                    "  (reads both CSVs from the given paths and updates dropdowns)",
                    style={"marginLeft": "8px", "fontSize": "0.9em"},
                ),
            ],
        ),
        html.Hr(),
        # Stores for parsed dataframes
        dcc.Store(id="store-data-1"),
        dcc.Store(id="store-data-2"),
        # Single graph with two vertically stacked panes
        html.Div(
            children=[
                html.H4("Series and Difference (x-axes linked)"),
                dcc.Graph(id="comparison-graph"),
            ],
            style={"marginBottom": "30px"},
        ),
    ],
)


def empty_figure(message: str = "Load two CSV files and select a series from each."):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=("Selected Series", "Difference (File 1 − File 2)"),
    )
    fig.update_layout(
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14),
            )
        ],
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(title_text="Index", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Difference", row=2, col=1)
    return fig


# ----------------------------------------------------------------------
# Load / reload CSVs from paths and update stores + dropdowns
# ----------------------------------------------------------------------
@app.callback(
    Output("store-data-1", "data"),
    Output("store-data-2", "data"),
    Output("column-dropdown-1", "options"),
    Output("column-dropdown-2", "options"),
    Output("file1-info", "children"),
    Output("file2-info", "children"),
    Input("reload-button", "n_clicks"),
    State("file1-path", "value"),
    State("file2-path", "value"),
    prevent_initial_call=True,
)
def load_files(n_clicks, path1, path2):
    df1_json = None
    df2_json = None
    options1 = []
    options2 = []
    info1 = ""
    info2 = ""

    # Normalize to str (State can return None)
    path1 = (path1 or "").strip()
    path2 = (path2 or "").strip()

    # Attempt to read file 1
    if path1:
        p1 = Path(path1)
        if p1.is_file():
            try:
                df1 = pd.read_csv(p1)
                df1_json = df1.to_json(date_format="iso", orient="split")
                numeric_cols1 = df1.select_dtypes(include="number").columns.tolist()
                options1 = [{"label": c, "value": c} for c in numeric_cols1]
                info1 = f"Loaded {p1} with {len(df1)} rows. Numeric columns: {', '.join(numeric_cols1) or 'None'}"
            except Exception as e:
                info1 = f"Failed to read {p1}: {e}"
        else:
            info1 = f"File not found: {p1}"
    else:
        info1 = "No path specified for File 1."

    # Attempt to read file 2
    if path2:
        p2 = Path(path2)
        if p2.is_file():
            try:
                df2 = pd.read_csv(p2)
                df2_json = df2.to_json(date_format="iso", orient="split")
                numeric_cols2 = df2.select_dtypes(include="number").columns.tolist()
                options2 = [{"label": c, "value": c} for c in numeric_cols2]
                info2 = f"Loaded {p2} with {len(df2)} rows. Numeric columns: {', '.join(numeric_cols2) or 'None'}"
            except Exception as e:
                info2 = f"Failed to read {p2}: {e}"
        else:
            info2 = f"File not found: {p2}"
    else:
        info2 = "No path specified for File 2."

    # Save config so paths persist across app restarts
    save_config(path1 if path1 else None, path2 if path2 else None)

    return df1_json, df2_json, options1, options2, info1, info2


# ----------------------------------------------------------------------
# Plot callback – uses stored data and selected columns
# ----------------------------------------------------------------------
@app.callback(
    Output("comparison-graph", "figure"),
    Input("store-data-1", "data"),
    Input("store-data-2", "data"),
    Input("column-dropdown-1", "value"),
    Input("column-dropdown-2", "value"),
)
def update_plots(data1, data2, col1, col2):
    if not data1 or not data2 or not col1 or not col2:
        return empty_figure()

    df1 = pd.read_json(data1, orient="split")
    df2 = pd.read_json(data2, orient="split")

    if col1 not in df1.columns or col2 not in df2.columns:
        return empty_figure("Selected columns not found in loaded data.")

    s1 = df1[col1].dropna().reset_index(drop=True)
    s2 = df2[col2].dropna().reset_index(drop=True)

    # Align lengths for comparison
    n = min(len(s1), len(s2))
    if n == 0:
        return empty_figure("No overlapping data between the two series.")

    s1 = s1.iloc[:n]
    s2 = s2.iloc[:n]
    x = list(range(n))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=("Selected Series", "Difference (File 1 − File 2)"),
    )

    # Top pane: both series
    fig.add_trace(
        go.Scatter(x=x, y=s1, mode="lines", name=f"File 1: {col1}"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=s2, mode="lines", name=f"File 2: {col2}"),
        row=1,
        col=1,
    )

    # Bottom pane: difference
    diff = s1 - s2
    fig.add_trace(
        go.Scatter(x=x, y=diff, mode="lines", name=f"{col1} − {col2}"),
        row=2,
        col=1,
    )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(title_text="Index", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Difference", row=2, col=1)

    return fig


if __name__ == "__main__":
    app.run(debug=True)
