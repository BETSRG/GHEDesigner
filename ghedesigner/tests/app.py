"""
Run with:
    pip install dash plotly pandas plotly
    python app.py

Features
--------
- Any number of panes (subplots), any number of series per pane.
- Fixed plot-area sizing: legend outside, fixed right margin.
- Live CSV polling + manual reload; zoom/pan preserved.
- Linked x-axis: zoom/pan any pane keeps all panes aligned.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, dcc, html, no_update
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------
# Data sources (edit paths as needed)
# ----------------------------------------------------------------------
# Defaults to the example CSVs placed next to this app.py.
HERE = Path(__file__).resolve().parent

DATA_FILES: dict[str, Path] = {
    "1-bldg, 1 GHE": HERE / "test_data" / "simulate_1_pipe_1_ghe_1_bldg_district.csv",
    "6-bldg, 3-GHE": HERE / "test_data" / "simulate_1_pipe_3_ghe_6_bldg_district_HOURLY.csv",
    "1-bldg, 1-GHE, 1-HX": HERE / "test_data" / "simulate_1_pipe_1_ghe_1_hx_1_bldg_district.csv",
    "1-bldg w/loads, 1-GHE, 1-HX": HERE / "test_data" / "simulate_1_pipe_1_ghe_1_hx_1_bldg_w_loads_district.csv",
}

X_COL = "Time [hr]"

CONTROL_CARD_STYLE = {
    "display": "flex",
    "flexDirection": "column",
    "gap": "0.5rem",
}

CONTROL_LABEL_STYLE = {
    "fontWeight": "600",
    "minHeight": "1.2rem",  # keeps labels aligned
}

CONTROL_BUTTON_STYLE = {
    "width": "100%",
    "height": "2.4rem",  # makes all buttons same height
}


def load_dataset(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Could not find {p}")
    df = pd.read_csv(p)
    if X_COL not in df.columns:
        raise ValueError(f"Missing required column '{X_COL}' in {p}")
    return df


# ----------------------------------------------------------------------
# Pane helpers
# ----------------------------------------------------------------------
def _category(col: str) -> str:
    """
    Heuristic grouping used ONLY for the initial default panes.
    """
    base = col.split(":", 1)[0] if ":" in col else col
    low = base.lower()
    if low.startswith("building"):
        return "Buildings"
    if low.startswith("ghe"):
        return "GHEs"
    if low.startswith("network"):
        return "Network"
    return base


def _metric(col: str) -> str:
    return col.split(":", 1)[1] if ":" in col else col


def default_panes(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Build default panes based on what's in the dataset:
      - Buildings pane: first building metric across all buildings
      - GHE pane: first ghe metric across all ghes
      - Network pane: first network metric
    """
    cols = [c for c in df.columns if c != X_COL]
    by_cat: dict[str, list[str]] = {}
    for c in cols:
        by_cat.setdefault(_category(c), []).append(c)

    panes: list[dict[str, Any]] = []

    if "Buildings" in by_cat:
        metrics = sorted({_metric(c) for c in by_cat["Buildings"] if ":" in c})
        if metrics:
            m0 = metrics[0]
            panes.append({"title": f"Buildings — {m0}", "columns": [c for c in by_cat["Buildings"] if c.endswith(m0)]})

    if "GHEs" in by_cat:
        metrics = sorted({_metric(c) for c in by_cat["GHEs"] if ":" in c})
        if metrics:
            m0 = metrics[0]
            panes.append({"title": f"GHEs — {m0}", "columns": [c for c in by_cat["GHEs"] if c.endswith(m0)]})

    if "Network" in by_cat:
        metrics = sorted({_metric(c) for c in by_cat["Network"] if ":" in c})
        if metrics:
            m0 = metrics[0]
            panes.append({"title": f"Network — {m0}", "columns": [c for c in by_cat["Network"] if c.endswith(m0)]})

    return panes or [{"title": "Pane 1", "columns": []}]


def sanitize_panes(panes: list[dict[str, Any]], available_cols: list[str]) -> list[dict[str, Any]]:
    avail = set(available_cols)
    out: list[dict[str, Any]] = []
    for i, p in enumerate(panes or []):
        out.append(
            {
                "title": str(p.get("title") or f"Pane {i + 1}"),
                "columns": [c for c in (p.get("columns") or []) if c in avail],
            }
        )
    return out or [{"title": "Pane 1", "columns": []}]


def parse_relayout(relayout: dict[str, Any]) -> tuple[list[Any] | None, dict[int, list[Any] | None]]:
    """
    Extract:
      - shared x range from xaxis.*
      - per-pane y ranges from yaxis, yaxis2, yaxis3, ...
    """
    x_range: list[Any] | None = None
    y_ranges: dict[int, list[Any] | None] = {}

    # X
    if "xaxis.range[0]" in relayout and "xaxis.range[1]" in relayout:
        x_range = [relayout["xaxis.range[0]"], relayout["xaxis.range[1]"]]
    elif relayout.get("xaxis.autorange"):
        x_range = None

    # Y (per subplot axis)
    for k, v in relayout.items():
        if not k.startswith("yaxis"):
            continue
        axis_part, rest = k.split(".", 1) if "." in k else (k, "")
        idx_str = axis_part.replace("yaxis", "")
        axis_idx = int(idx_str) if idx_str else 1

        if rest.startswith("range[0]"):
            hi_key = f"{axis_part}.range[1]"
            if hi_key in relayout:
                y_ranges[axis_idx] = [v, relayout[hi_key]]
        elif rest == "autorange" and bool(v):
            y_ranges[axis_idx] = None

    return x_range, y_ranges


def build_figure(df: pd.DataFrame, panes: list[dict[str, Any]], axis_state: dict[str, Any] | None) -> go.Figure:
    axis_state = axis_state or {"x": None, "y": {}}
    x_range = axis_state.get("x")
    y_ranges: dict[int, list[Any] | None] = axis_state.get("y", {}) or {}

    n = max(1, len(panes))
    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[p.get("title", f"Pane {i + 1}") for i, p in enumerate(panes)],
    )

    x = df[X_COL]

    for row, pane in enumerate(panes, start=1):
        cols = pane.get("columns", []) or []
        if not cols:
            fig.add_trace(
                go.Scatter(x=x, y=[None] * len(df), mode="lines", showlegend=False),
                row=row,
                col=1,
            )
            fig.add_annotation(
                x=0.5,
                y=0.5,
                xref=f"x{'' if row == 1 else row} domain",
                yref=f"y{'' if row == 1 else row} domain",
                text="No series selected for this pane",
                showarrow=False,
                font={"color": "#888"},
            )
            continue

        for c in cols:
            fig.add_trace(go.Scatter(x=x, y=df[c], mode="lines", name=c), row=row, col=1)

    # Apply persisted ranges
    if isinstance(x_range, list) and len(x_range) == 2:
        fig.update_xaxes(range=x_range)
    for row in range(1, n + 1):
        yr = y_ranges.get(row)
        if isinstance(yr, list) and len(yr) == 2:
            fig.update_yaxes(range=yr, row=row, col=1)

    # Fixed right margin prevents legend size from resizing plot areas.
    fig.update_layout(
        margin={"l": 60, "r": 260, "t": 60, "b": 50},
        legend={
            "orientation": "v",
            "x": 1.02,
            "y": 1.0,
            "xanchor": "left",
            "yanchor": "top",
            "title": {"text": "Series"},
        },
        hovermode="x unified",
        uirevision="keep",  # critical: preserves zoom/pan across updates
        height=260 * n + 80,
    )
    fig.update_xaxes(title_text="Hour", row=n, col=1)
    return fig


# ----------------------------------------------------------------------
# Dash app
# ----------------------------------------------------------------------
app = Dash(__name__)
app.title = "District Time-Series Dashboard"

app.layout = html.Div(
    style={"fontFamily": "system-ui, sans-serif", "margin": "20px"},
    children=[
        dcc.Store(id="datasets-store"),
        dcc.Store(id="columns-store"),
        dcc.Store(id="panes-store"),
        dcc.Store(id="axis-store"),
        dcc.Interval(id="poll-interval", interval=300000, n_intervals=0),  # 2s polling
        html.H1("District Time-Series Dashboard", style={"marginBottom": "0.25rem"}),
        html.P("Multi-pane time-series explorer (linked x-axis, live reload).", style={"color": "#555"}),
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(4, minmax(240px, 1fr))",
                "gap": "1rem",
                "alignItems": "end",
                "marginBottom": "0.75rem",
            },
            children=[
                html.Div(
                    children=[
                        html.Label("Dataset", style={"fontWeight": "600"}),
                        dcc.Dropdown(
                            id="dataset-dropdown",
                            options=[{"label": name, "value": name} for name in DATA_FILES],
                            value=next(iter(DATA_FILES.keys())),
                            clearable=False,
                        ),
                    ]
                ),
                html.Div(
                    style=CONTROL_CARD_STYLE,
                    children=[
                        html.Label("Panes", style=CONTROL_LABEL_STYLE),
                        html.Button("Add pane", id="add-pane", n_clicks=0, style=CONTROL_BUTTON_STYLE),
                        html.Button("Remove pane", id="remove-pane", n_clicks=0, style=CONTROL_BUTTON_STYLE),
                        html.Div(" ", style={"fontSize": "0.9rem", "minHeight": "1.1rem"}),  # spacer to match others
                    ],
                ),
                html.Div(
                    style=CONTROL_CARD_STYLE,
                    children=[
                        html.Label("Data", style=CONTROL_LABEL_STYLE),
                        html.Button("Reload CSV files now", id="reload-button", n_clicks=0, style=CONTROL_BUTTON_STYLE),
                        html.Div(
                            "Polled every 300 seconds.",
                            style={"color": "#666", "fontSize": "0.9rem", "minHeight": "1.1rem"},
                        ),
                    ],
                ),
                html.Div(
                    style=CONTROL_CARD_STYLE,
                    children=[
                        html.Label("Reset view", style=CONTROL_LABEL_STYLE),
                        html.Button("Reset zoom/pan", id="reset-view", n_clicks=0, style=CONTROL_BUTTON_STYLE),
                        html.Div(" ", style={"fontSize": "0.9rem", "minHeight": "1.1rem"}),  # spacer to match others
                    ],
                ),
            ],
        ),
        html.Div(id="reload-status", style={"color": "#555", "marginBottom": "0.75rem"}),
        html.Hr(style={"margin": "1rem 0"}),
        html.Div(id="pane-controls", style={"display": "flex", "flexDirection": "column", "gap": "0.75rem"}),
        html.Hr(style={"margin": "1rem 0"}),
        dcc.Graph(id="main-graph", style={"height": "700px"}),
    ],
)


# ----------------------------------------------------------------------
# Data loading (poll + manual reload)
# ----------------------------------------------------------------------
@app.callback(
    Output("datasets-store", "data"),
    Output("columns-store", "data"),
    Output("reload-status", "children"),
    Input("poll-interval", "n_intervals"),
    Input("reload-button", "n_clicks"),
    prevent_initial_call=False,
)
def load_all(_n_intervals: int, _n_clicks: int):
    datasets: dict[str, list[dict[str, Any]]] = {}
    columns: dict[str, list[str]] = {}
    try:
        for name, path in DATA_FILES.items():
            df = load_dataset(path)
            datasets[name] = [{str(k): v for k, v in record.items()} for record in df.to_dict("records")]
            columns[name] = [c for c in df.columns if c != X_COL]
    except (FileNotFoundError, OSError, ValueError, pd.errors.ParserError) as exc:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return no_update, no_update, f"Reload failed at {ts}: {exc}"

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return datasets, columns, f"Loaded/updated datasets at {ts}."


# ----------------------------------------------------------------------
# Initialize panes on first load / sanitize on dataset change
# ----------------------------------------------------------------------
@app.callback(
    Output("panes-store", "data"),
    Input("dataset-dropdown", "value"),
    Input("datasets-store", "data"),
    State("panes-store", "data"),
    prevent_initial_call=False,
)
def init_or_sanitize_panes(dataset: str, ds_store: dict[str, Any] | None, panes_state: Any):
    if not ds_store or dataset not in ds_store:
        return panes_state or [{"title": "Pane 1", "columns": []}]
    df = pd.DataFrame(ds_store[dataset])
    avail = [c for c in df.columns if c != X_COL]

    if isinstance(panes_state, list) and panes_state:
        return sanitize_panes(panes_state, avail)

    return default_panes(df)


# ----------------------------------------------------------------------
# Add/remove panes
# ----------------------------------------------------------------------
@app.callback(
    Output("panes-store", "data", allow_duplicate=True),
    Input("add-pane", "n_clicks"),
    Input("remove-pane", "n_clicks"),
    State("panes-store", "data"),
    prevent_initial_call=True,
)
def edit_panes(_add: int, _remove: int, panes: list[dict[str, Any]] | None):
    panes = list(panes or [{"title": "Pane 1", "columns": []}])
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    if trig == "add-pane":
        panes.append({"title": f"Pane {len(panes) + 1}", "columns": []})
    elif trig == "remove-pane" and len(panes) > 1:
        panes.pop()

    return panes


# ----------------------------------------------------------------------
# Pane controls UI (dynamic components)
# ----------------------------------------------------------------------
@app.callback(
    Output("pane-controls", "children"),
    Input("dataset-dropdown", "value"),
    Input("panes-store", "data"),
    Input("columns-store", "data"),
)
def render_controls(dataset: str, panes: list[dict[str, Any]] | None, col_store: dict[str, Any] | None):
    panes = panes or [{"title": "Pane 1", "columns": []}]
    cols = (col_store or {}).get(dataset, []) or []
    options = [{"label": c, "value": c} for c in cols]

    children: list[Any] = []
    for i, p in enumerate(panes):
        children.append(
            html.Div(
                style={
                    "border": "1px solid #ddd",
                    "borderRadius": "8px",
                    "padding": "0.75rem",
                    "background": "#fafafa",
                },
                children=[
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "240px 1fr", "gap": "0.75rem"},
                        children=[
                            html.Div(
                                children=[
                                    html.Label(f"Pane {i + 1} title", style={"fontWeight": "600"}),
                                    dcc.Input(
                                        id={"type": "pane-title", "index": i},
                                        value=p.get("title", f"Pane {i + 1}"),
                                        type="text",
                                        debounce=True,
                                        style={"width": "100%"},
                                    ),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Label("Series (columns)", style={"fontWeight": "600"}),
                                    dcc.Dropdown(
                                        id={"type": "pane-columns", "index": i},
                                        options=options,
                                        value=p.get("columns", []),
                                        multi=True,
                                        placeholder="Select one or more columns…",
                                    ),
                                ]
                            ),
                        ],
                    )
                ],
            )
        )
    return children


@app.callback(
    Output("panes-store", "data", allow_duplicate=True),
    Input({"type": "pane-title", "index": ALL}, "value"),
    Input({"type": "pane-columns", "index": ALL}, "value"),
    State("panes-store", "data"),
    prevent_initial_call=True,
)
def update_panes_store(
    titles: list[str], columns: list[list[str]], panes: list[dict[str, str | list[str]]]
) -> list[dict[str, str | list[str]]]:
    panes = list(panes or [])
    if not panes:
        return no_update

    n = len(panes)
    resized_titles = (titles or [])[:n] + [None] * max(0, n - len(titles or []))
    resized_columns = (columns or [])[:n] + [None] * max(0, n - len(columns or []))

    out: list[dict[str, str | list[str]]] = []
    for i in range(n):
        out.append(
            {
                "title": resized_titles[i] or panes[i].get("title", f"Pane {i + 1}"),
                "columns": resized_columns[i] or panes[i].get("columns", []),
            }
        )
    return out


# ----------------------------------------------------------------------
# Axis sync store
# ----------------------------------------------------------------------
@app.callback(
    Output("axis-store", "data"),
    Input("main-graph", "relayoutData"),
    Input("reset-view", "n_clicks"),
    State("axis-store", "data"),
    prevent_initial_call=True,
)
def sync_axes(relayout: dict[str, Any] | None, _reset: int, axis: dict[str, Any] | None):
    axis = axis or {"x": None, "y": {}}
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    if trig == "reset-view":
        return {"x": None, "y": {}}

    if not relayout:
        return no_update

    x_new, y_new = parse_relayout(relayout)
    x_range_out = axis.get("x")
    y_ranges_dict = dict(axis.get("y", {}) or {})

    # update x if relayout touched x
    if "xaxis.autorange" in relayout or ("xaxis.range[0]" in relayout and "xaxis.range[1]" in relayout):
        x_range_out = x_new

    # update only y axes mentioned
    for idx, yr in y_new.items():
        y_ranges_dict[idx] = yr

    return {"x": x_range_out, "y": y_ranges_dict}


# ----------------------------------------------------------------------
# Main figure callback
# ----------------------------------------------------------------------
@app.callback(
    Output("main-graph", "figure"),
    Input("dataset-dropdown", "value"),
    Input("datasets-store", "data"),
    Input("panes-store", "data"),
    Input("axis-store", "data"),
)
def update_figure(
    dataset: str, ds_store: dict[str, Any] | None, panes: list[dict[str, Any]] | None, axis: dict[str, Any] | None
):
    if not ds_store or dataset not in ds_store:
        fig = go.Figure()
        fig.update_layout(title="No data loaded")
        return fig

    df = pd.DataFrame(ds_store[dataset])
    panes = panes or [{"title": "Pane 1", "columns": []}]
    panes = sanitize_panes(panes, [c for c in df.columns if c != X_COL])

    return build_figure(df, panes, axis)


if __name__ == "__main__":
    app.run(debug=True)
