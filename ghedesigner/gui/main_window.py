from __future__ import annotations

import ctypes
import json
import sys
from contextlib import suppress
from functools import partial
from pathlib import Path
from threading import Thread
from tkinter import (
    BOTH,
    END,
    HORIZONTAL,
    LEFT,
    RIGHT,
    VERTICAL,
    Canvas,
    Entry,
    Frame,
    Label,
    LabelFrame,
    Menu,
    Misc,
    PanedWindow,
    Scrollbar,
    StringVar,
    Text,
    Tk,
    filedialog,
    font,
    messagebox,
    ttk,
)
from tkinter.constants import LAST
from typing import Any

from jsonschema import Draft7Validator

from ghedesigner.gui.ghedesigner_adapter import (
    NetworkValidationError,
    export_to_ghedesigner,
    validate_file_paths,
    validate_network,
)
from ghedesigner.gui.models import COMPONENT_TYPES, NODE_COLORS, ComponentType, NetworkDocument
from ghedesigner.gui.run_paths import build_run_paths
from ghedesigner.main import run as run_ghedesigner

NODE_WIDTH = 180
NODE_HEIGHT = 66
NODE_RADIUS = 10


def _enable_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    with suppress(Exception):
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    with suppress(Exception):
        ctypes.windll.user32.SetProcessDPIAware()


_enable_dpi_awareness()


class GHEDesignerWindow(Tk):
    """Draft drag-and-drop editor for GHEDesigner thermal network inputs."""

    def __init__(self) -> None:
        super().__init__(className="GHEDesignerWindow")
        self.title("GHEDesigner Network Editor")
        self.geometry("1220x760")
        self.minsize(1000, 620)
        self._configure_fonts()

        self.document = NetworkDocument.starter()
        self.selected_node_id: str | None = None
        self.connect_source_id: str | None = None
        self.drag_node_id: str | None = None
        self.drag_last_xy: tuple[int, int] | None = None
        self.status_var = StringVar(value="Ready")
        self.name_var = StringVar()
        self.type_var = StringVar(value="No component selected")
        self.upstream_var = StringVar(value="Upstream: -")
        self.downstream_var = StringVar(value="Downstream: -")
        self.output_dir_var = StringVar(value=str(Path.cwd() / "tmp" / "ghedesigner-gui"))
        self.simulation_running = False
        self.node_widgets: dict[str, Frame] = {}
        self.node_window_items: dict[str, int] = {}

        self._build_menu()
        self._build_layout()
        self._bind_shortcuts()
        self._redraw()

    def _configure_fonts(self) -> None:
        with suppress(Exception):
            self.tk.call("tk", "scaling", max(1.0, min(2.0, self.winfo_fpixels("1i") / 72.0)))

        families = set(font.families(self))
        preferred_family = next(
            (family for family in ("DejaVu Sans", "Noto Sans", "Liberation Sans", "Arial") if family in families),
            "TkDefaultFont",
        )
        font_specs = {
            "TkDefaultFont": 10,
            "TkTextFont": 10,
            "TkMenuFont": 10,
            "TkHeadingFont": 11,
            "TkCaptionFont": 10,
            "TkSmallCaptionFont": 9,
            "TkIconFont": 10,
            "TkTooltipFont": 9,
        }
        for font_name, size in font_specs.items():
            with suppress(Exception):
                named_font = font.nametofont(font_name)
                named_font.configure(family=preferred_family, size=size)

        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TButton", padding=(8, 5), font=(preferred_family, 10))
        style.configure("TLabel", font=(preferred_family, 10))
        style.configure("TLabelframe.Label", font=(preferred_family, 10, "bold"))
        self.option_add("*Font", "TkDefaultFont")

    def _build_menu(self) -> None:
        menubar = Menu(self)
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="New Project", command=self._new_project)
        file_menu.add_command(label="Open Project...", command=self._load_project)
        file_menu.add_command(label="Save Project...", command=self._save_project)
        file_menu.add_separator()
        file_menu.add_command(label="Export GHEDesigner JSON...", command=self._export_ghedesigner_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)

    def _build_layout(self) -> None:
        root = PanedWindow(self, orient=HORIZONTAL, sashrelief="raised")
        root.pack(fill=BOTH, expand=True)

        palette = Frame(root, padx=8, pady=8)
        root.add(palette, minsize=190)
        self._build_palette(palette)

        center = Frame(root)
        root.add(center, minsize=500)
        self._build_canvas(center)

        inspector = Frame(root, padx=8, pady=8)
        root.add(inspector, minsize=340)
        self._build_inspector(inspector)

        Label(self, textvariable=self.status_var, anchor="w", relief="sunken", padx=8).pack(fill="x", side="bottom")

    def _build_palette(self, parent: Frame) -> None:
        ttk.Label(parent, text="Components", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(0, 8))
        for component_type, label in COMPONENT_TYPES.items():
            ttk.Button(parent, text=f"Add {label}", command=partial(self._add_component, component_type)).pack(
                fill="x", pady=3
            )

        ttk.Label(parent, text="Workflow", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(18, 8))
        ttk.Button(parent, text="Set selected as upstream", command=self._begin_connection).pack(fill="x", pady=3)
        ttk.Button(parent, text="Cancel connection", command=self._cancel_connection).pack(fill="x", pady=3)
        ttk.Button(parent, text="Delete selected", command=self._delete_selected).pack(fill="x", pady=3)
        ttk.Button(parent, text="Clear connections", command=self._clear_connections).pack(fill="x", pady=3)
        ttk.Button(parent, text="Validate export", command=self._show_validation_dialog).pack(fill="x", pady=3)

        ttk.Label(parent, text="Run", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(18, 8))
        Entry(parent, textvariable=self.output_dir_var).pack(fill="x", pady=(0, 4))
        ttk.Button(parent, text="Select output folder", command=self._select_output_folder).pack(fill="x", pady=3)
        self.execute_button = ttk.Button(parent, text="Execute simulation", command=self._execute_simulation)
        self.execute_button.pack(fill="x", pady=3)

        ttk.Label(
            parent,
            text=(
                "Connections are ordered loop links. Select a component, click "
                "Set selected as upstream, then click its downstream component. "
                "Existing upstream/downstream links are replaced."
            ),
            justify=LEFT,
            wraplength=170,
        ).pack(anchor="w", pady=(18, 0))

    def _build_canvas(self, parent: Frame) -> None:
        toolbar = Frame(parent, padx=8, pady=6)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Fit starter layout", command=self._fit_starter_layout).pack(side=LEFT)
        ttk.Button(toolbar, text="Export preview", command=self._refresh_preview).pack(side=LEFT, padx=(8, 0))

        self.canvas = Canvas(parent, bg="#f8fafc", highlightthickness=0)
        self.canvas.pack(fill=BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

    def _build_inspector(self, parent: Frame) -> None:
        details = LabelFrame(parent, text="Selected component", padx=6, pady=6)
        details.pack(fill="x")
        Label(details, text="Name").grid(row=0, column=0, sticky="w")
        Entry(details, textvariable=self.name_var).grid(row=1, column=0, sticky="ew", pady=(0, 6))
        Label(details, textvariable=self.type_var, fg="#475569").grid(row=2, column=0, sticky="w")
        Label(details, textvariable=self.upstream_var, fg="#475569").grid(row=3, column=0, sticky="w", pady=(6, 0))
        Label(details, textvariable=self.downstream_var, fg="#475569").grid(row=4, column=0, sticky="w")
        details.grid_columnconfigure(0, weight=1)

        ttk.Button(details, text="Apply component JSON", command=self._apply_component_edits).grid(
            row=5, column=0, sticky="ew", pady=(6, 0)
        )

        config_frame = LabelFrame(parent, text="Component JSON", padx=6, pady=6)
        config_frame.pack(fill=BOTH, expand=True, pady=(8, 0))
        self.config_text = self._scrolling_text(config_frame, height=14)

        preview_frame = LabelFrame(parent, text="GHEDesigner export preview", padx=6, pady=6)
        preview_frame.pack(fill=BOTH, expand=True, pady=(8, 0))
        self.preview_text = self._scrolling_text(preview_frame, height=14)

    def _scrolling_text(self, parent: Misc, height: int) -> Text:
        frame = Frame(parent)
        frame.pack(fill=BOTH, expand=True)
        scrollbar = Scrollbar(frame, orient=VERTICAL)
        text = Text(frame, height=height, wrap="none", yscrollcommand=scrollbar.set, undo=True)
        scrollbar.config(command=text.yview)
        scrollbar.pack(side=RIGHT, fill="y")
        text.pack(side=LEFT, fill=BOTH, expand=True)
        return text

    def _bind_shortcuts(self) -> None:
        self.bind("<Control-n>", lambda _event: self._new_project())
        self.bind("<Control-s>", lambda _event: self._save_project())
        self.bind("<Delete>", lambda _event: self._delete_selected())
        self.bind("<Escape>", lambda _event: self._cancel_connection())

    def _add_component(self, component_type: ComponentType) -> None:
        offset = len(self.document.nodes) * 28
        node = self.document.add_node(component_type, 180 + offset, 180 + offset)
        self.selected_node_id = node.id
        self._set_status(f"Added {node.name}")
        self._redraw()

    def _begin_connection(self) -> None:
        selected = self.document.find_node(self.selected_node_id)
        if selected is None:
            self._set_status("Select a component before beginning a connection.")
            return
        self.connect_source_id = selected.id
        self._set_status(
            f"Set upstream to {selected.name}. Click its downstream component; existing links will be replaced."
        )
        self._redraw()

    def _cancel_connection(self) -> None:
        self.connect_source_id = None
        self._set_status("Connection canceled.")
        self._redraw()

    def _delete_selected(self) -> None:
        selected = self.document.find_node(self.selected_node_id)
        if selected is None:
            return
        self.document.delete_node(selected.id)
        self.selected_node_id = None
        self.connect_source_id = None
        self._set_status(f"Deleted {selected.name}")
        self._redraw()

    def _clear_connections(self) -> None:
        self.document.edges.clear()
        self.connect_source_id = None
        self._set_status("Connections cleared.")
        self._redraw()

    def _fit_starter_layout(self) -> None:
        type_offsets = {
            "building": (120, 150),
            "isolated_horizontal_pipe": (340, 150),
            "coupled_horizontal_pipe": (340, 260),
            "source_sink_heat_exchanger": (560, 150),
            "ground_heat_exchanger": (780, 150),
        }
        counts: dict[str, int] = {}
        for node in self.document.nodes:
            base_x, base_y = type_offsets.get(node.component_type, (120, 360))
            count = counts.get(node.component_type, 0)
            counts[node.component_type] = count + 1
            node.x = base_x
            node.y = base_y + count * 92
        self._redraw()

    def _on_canvas_press(self, event: Any) -> None:
        node_id = self._node_id_at(event.x, event.y)
        if node_id is None:
            self.selected_node_id = None
            self.drag_node_id = None
            self._redraw()
            return
        self._handle_node_press(node_id, event.x, event.y)

    def _on_node_widget_press(self, event: Any, node_id: str) -> None:
        canvas_x, canvas_y = self._event_canvas_xy(event)
        self._handle_node_press(node_id, canvas_x, canvas_y)

    def _handle_node_press(self, node_id: str, canvas_x: int, canvas_y: int) -> None:
        if self.connect_source_id and self.connect_source_id != node_id:
            try:
                self.document.set_downstream(self.connect_source_id, node_id)
                source = self.document.find_node(self.connect_source_id)
                target = self.document.find_node(node_id)
                if source is not None and target is not None:
                    self._set_status(f"Set downstream: {source.name} -> {target.name}")
            except ValueError as error:
                self._set_status(str(error))
            self.connect_source_id = None
            self.selected_node_id = node_id
            self._redraw()
            return

        self.selected_node_id = node_id
        self.drag_node_id = node_id
        self.drag_last_xy = (canvas_x, canvas_y)
        self._redraw()

    def _on_canvas_drag(self, event: Any) -> None:
        self._drag_selected_node(event.x, event.y)

    def _drag_selected_node(self, canvas_x: int, canvas_y: int) -> None:
        node = self.document.find_node(self.drag_node_id)
        if node is None or self.drag_last_xy is None:
            return
        last_x, last_y = self.drag_last_xy
        node.x += canvas_x - last_x
        node.y += canvas_y - last_y
        self.drag_last_xy = (canvas_x, canvas_y)
        item_id = self.node_window_items.get(node.id)
        if item_id is not None:
            self.canvas.coords(item_id, node.x, node.y)
            self._redraw_edges()
        else:
            self._redraw(update_preview=False)

    def _on_node_widget_drag(self, event: Any) -> None:
        canvas_x, canvas_y = self._event_canvas_xy(event)
        self._drag_selected_node(canvas_x, canvas_y)

    def _on_node_widget_release(self, _event: Any) -> None:
        self.drag_node_id = None
        self.drag_last_xy = None
        self._refresh_preview()

    def _on_canvas_release(self, _event: Any) -> None:
        self.drag_node_id = None
        self.drag_last_xy = None
        self._refresh_preview()

    def _event_canvas_xy(self, event: Any) -> tuple[int, int]:
        return (event.x_root - self.canvas.winfo_rootx(), event.y_root - self.canvas.winfo_rooty())

    def _node_id_at(self, x: int, y: int) -> str | None:
        for node in reversed(self.document.nodes):
            if node.x <= x <= node.x + NODE_WIDTH and node.y <= y <= node.y + NODE_HEIGHT:
                return node.id
        return None

    def _apply_component_edits(self) -> None:
        if self._sync_selected_component_edits():
            node = self.document.find_node(self.selected_node_id)
            if node is not None:
                self._set_status(f"Updated {node.name}")
            self._redraw()

    def _sync_selected_component_edits(self) -> bool:
        node = self.document.find_node(self.selected_node_id)
        if node is None:
            return True
        name = self.name_var.get().strip()
        if not name:
            self._set_status("Component name cannot be blank.")
            return False
        try:
            config = json.loads(self.config_text.get("1.0", END))
        except json.JSONDecodeError as error:
            self._set_status(f"Invalid component JSON: {error}")
            return False
        if not isinstance(config, dict):
            self._set_status("Component JSON must be an object.")
            return False
        node.name = name
        node.config = config
        return True

    def _redraw(self, update_preview: bool = True) -> None:
        for widget in self.node_widgets.values():
            widget.destroy()
        self.node_widgets.clear()
        self.node_window_items.clear()
        self.canvas.delete("all")
        self._draw_grid()
        for edge in self.document.edges:
            source = self.document.find_node(edge.source)
            target = self.document.find_node(edge.target)
            if source and target:
                self._draw_edge(source.x, source.y, target.x, target.y)
        for node in self.document.nodes:
            self._draw_node(node)
        self._refresh_inspector()
        if update_preview:
            self._refresh_preview()

    def _draw_grid(self) -> None:
        width = max(self.canvas.winfo_width(), 1200)
        height = max(self.canvas.winfo_height(), 800)
        for x in range(0, width, 32):
            self.canvas.create_line(x, 0, x, height, fill="#e2e8f0")
        for y in range(0, height, 32):
            self.canvas.create_line(0, y, width, y, fill="#e2e8f0")

    def _draw_edge(self, source_x: float, source_y: float, target_x: float, target_y: float) -> None:
        x1 = source_x + NODE_WIDTH / 2
        y1 = source_y + NODE_HEIGHT / 2
        x2 = target_x + NODE_WIDTH / 2
        y2 = target_y + NODE_HEIGHT / 2
        self.canvas.create_line(
            x1, y1, x2, y2, width=2, fill="#334155", arrow=LAST, arrowshape=(12, 14, 5), tags=("edge",)
        )

    def _redraw_edges(self) -> None:
        self.canvas.delete("edge")
        for edge in self.document.edges:
            source = self.document.find_node(edge.source)
            target = self.document.find_node(edge.target)
            if source and target:
                self._draw_edge(source.x, source.y, target.x, target.y)
        self.canvas.tag_lower("edge")

    def _draw_node(self, node: Any) -> None:
        selected = node.id == self.selected_node_id
        source = node.id == self.connect_source_id
        outline = "#0891b2" if source else "#0f172a" if selected else "#94a3b8"
        width = 2 if selected or source else 1
        fill = "#ecfeff" if source else "#ffffff"
        color = NODE_COLORS.get(node.component_type, "#475569")

        self.canvas.create_rectangle(
            node.x + 3,
            node.y + 4,
            node.x + NODE_WIDTH + 3,
            node.y + NODE_HEIGHT + 4,
            fill="#cbd5e1",
            outline="",
        )
        frame = Frame(
            self.canvas,
            bg=fill,
            highlightbackground=outline,
            highlightcolor=outline,
            highlightthickness=width,
            width=NODE_WIDTH,
            height=NODE_HEIGHT,
            cursor="fleur",
        )
        frame.pack_propagate(False)
        strip = Frame(frame, bg=color, width=NODE_RADIUS)
        strip.pack(side=LEFT, fill="y")
        content = Frame(frame, bg=fill, padx=8, pady=6)
        content.pack(side=LEFT, fill=BOTH, expand=True)
        Label(
            content,
            text=node.name,
            anchor="w",
            bg=fill,
            fg="#0f172a",
            font=("TkDefaultFont", 10, "bold"),
            wraplength=NODE_WIDTH - 32,
            justify=LEFT,
        ).pack(fill="x")
        Label(
            content,
            text=COMPONENT_TYPES.get(node.component_type, node.component_type),
            anchor="w",
            bg=fill,
            fg="#475569",
            font=("TkDefaultFont", 9),
            wraplength=NODE_WIDTH - 32,
            justify=LEFT,
        ).pack(fill="x", pady=(3, 0))

        self._bind_node_widget(frame, node.id)
        for child in frame.winfo_children():
            self._bind_node_widget(child, node.id)
            for grandchild in child.winfo_children():
                self._bind_node_widget(grandchild, node.id)

        self.node_widgets[node.id] = frame
        item_id = self.canvas.create_window(
            node.x,
            node.y,
            anchor="nw",
            window=frame,
            width=NODE_WIDTH,
            height=NODE_HEIGHT,
            tags=("node", f"node:{node.id}"),
        )
        self.node_window_items[node.id] = item_id

    def _bind_node_widget(self, widget: Any, node_id: str) -> None:
        widget.bind("<ButtonPress-1>", lambda event, current=node_id: self._on_node_widget_press(event, current))
        widget.bind("<B1-Motion>", self._on_node_widget_drag)
        widget.bind("<ButtonRelease-1>", self._on_node_widget_release)

    def _refresh_inspector(self) -> None:
        node = self.document.find_node(self.selected_node_id)
        if node is None:
            self.name_var.set("")
            self.type_var.set("No component selected")
            self.upstream_var.set("Upstream: -")
            self.downstream_var.set("Downstream: -")
            self._replace_text(self.config_text, "")
            return
        upstream = next((edge.source for edge in self.document.edges if edge.target == node.id), None)
        downstream = next((edge.target for edge in self.document.edges if edge.source == node.id), None)
        self.name_var.set(node.name)
        self.type_var.set(COMPONENT_TYPES.get(node.component_type, node.component_type))
        self.upstream_var.set(f"Upstream: {self._node_name(upstream)}")
        self.downstream_var.set(f"Downstream: {self._node_name(downstream)}")
        self._replace_text(self.config_text, json.dumps(node.config, indent=2))

    def _node_name(self, node_id: str | None) -> str:
        node = self.document.find_node(node_id)
        return node.name if node else "-"

    def _refresh_preview(self) -> None:
        try:
            exported = export_to_ghedesigner(self.document)
            preview = json.dumps(exported, indent=2)
            schema_messages = self._validate_export_schema(exported)
            path_messages = validate_file_paths(exported)
            warning_messages = schema_messages + ["Missing file path: " + message for message in path_messages]
            if warning_messages:
                preview = "Warnings:\n" + "\n".join(warning_messages) + "\n\n" + preview
        except NetworkValidationError as error:
            preview = "Export is not ready:\n" + str(error) + "\n\nProject document:\n"
            preview += json.dumps(self.document.to_dict(), indent=2)
        self._replace_text(self.preview_text, preview)

    def _validate_export_schema(self, exported: dict[str, Any]) -> list[str]:
        schema_path = Path(__file__).parents[1] / "schemas" / "ghedesigner.schema.json"
        schema = json.loads(schema_path.read_text())
        errors = sorted(Draft7Validator(schema).iter_errors(exported), key=lambda error: list(error.path))
        return [f"/{'/'.join(str(part) for part in error.path)}: {error.message}" for error in errors[:4]]

    def _replace_text(self, widget: Text, value: str) -> None:
        widget.delete("1.0", END)
        widget.insert("1.0", value)

    def _new_project(self) -> None:
        self.document = NetworkDocument.starter()
        self.selected_node_id = None
        self.connect_source_id = None
        self._set_status("New project created.")
        self._redraw()

    def _save_project(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".ghed-network.json",
            filetypes=[("GHEDesigner network project", "*.ghed-network.json"), ("JSON", "*.json")],
        )
        if not path:
            return
        Path(path).write_text(json.dumps(self.document.to_dict(), indent=2))
        self._set_status(f"Saved project to {path}")

    def _load_project(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All files", "*")])
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text())
            self.document = NetworkDocument.from_dict(data)
        except (OSError, json.JSONDecodeError, TypeError) as error:
            messagebox.showerror("Open failed", str(error))
            return
        self.selected_node_id = None
        self.connect_source_id = None
        self._set_status(f"Loaded project from {path}")
        self._redraw()

    def _select_output_folder(self) -> None:
        initial_dir = self.output_dir_var.get().strip() or str(Path.cwd())
        path = filedialog.askdirectory(initialdir=initial_dir, mustexist=False)
        if path:
            self.output_dir_var.set(path)

    def _execute_simulation(self) -> None:
        if self.simulation_running:
            self._set_status("A simulation is already running.")
            return

        output_dir_text = self.output_dir_var.get().strip()
        if not output_dir_text:
            messagebox.showerror("Output folder required", "Select an output folder before running a simulation.")
            return

        if not self._sync_selected_component_edits():
            return

        try:
            exported = export_to_ghedesigner(self.document)
        except NetworkValidationError as error:
            messagebox.showerror("Simulation input is not ready", str(error))
            return

        schema_messages = self._validate_export_schema(exported)
        if schema_messages:
            messagebox.showerror("Simulation input is not schema-valid", "\n".join(schema_messages))
            return

        path_messages = validate_file_paths(exported)
        if path_messages:
            messagebox.showerror("Simulation input references missing files", "\n".join(path_messages))
            return

        output_parent = Path(output_dir_text).expanduser().resolve()
        output_dir, input_path, run_stem = build_run_paths(output_parent, self.document.title)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            input_path.write_text(json.dumps(exported, indent=2))
        except OSError as error:
            messagebox.showerror("Output folder error", str(error))
            return

        self.simulation_running = True
        self.execute_button.configure(state="disabled")
        self._set_status(f"Running {run_stem} with {input_path} ...")
        Thread(target=self._run_simulation_worker, args=(input_path, output_dir), daemon=True).start()

    def _run_simulation_worker(self, input_path: Path, output_dir: Path) -> None:
        try:
            return_code = run_ghedesigner(input_path, output_dir)
        except Exception as error:  # noqa: BLE001
            self.after(0, self._simulation_finished, 1, str(error), output_dir)
            return
        self.after(0, self._simulation_finished, return_code, "", output_dir)

    def _simulation_finished(self, return_code: int, error_message: str, output_dir: Path) -> None:
        self.simulation_running = False
        self.execute_button.configure(state="normal")
        if return_code == 0:
            self._set_status(f"Simulation complete. Outputs written to {output_dir}")
            messagebox.showinfo("Simulation complete", f"Outputs written to:\n{output_dir}")
        else:
            message = error_message or f"GHEDesigner returned code {return_code}."
            self._set_status("Simulation failed.")
            messagebox.showerror("Simulation failed", message)

    def _export_ghedesigner_file(self) -> None:
        if not self._sync_selected_component_edits():
            return
        try:
            exported = export_to_ghedesigner(self.document)
        except NetworkValidationError as error:
            messagebox.showerror("Export is not ready", str(error))
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".ghedesigner.json",
            filetypes=[("GHEDesigner input", "*.ghedesigner.json"), ("JSON", "*.json")],
        )
        if not path:
            return
        Path(path).write_text(json.dumps(exported, indent=2))
        self._set_status(f"Exported GHEDesigner JSON to {path}")

    def _show_validation_dialog(self) -> None:
        messages = validate_network(self.document)
        if not messages:
            try:
                exported = export_to_ghedesigner(self.document)
                schema_messages = self._validate_export_schema(exported)
                path_messages = ["Missing file path: " + message for message in validate_file_paths(exported)]
            except NetworkValidationError as error:
                schema_messages = [str(error)]
                path_messages = []
            messages = schema_messages + path_messages or ["The current network is ready to export."]
        messagebox.showinfo("Validation", "\n".join(messages))

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)

    def run(self) -> None:
        self.mainloop()
