from enum import Enum, auto
from json import dumps, loads
from pathlib import Path
from queue import Queue
from tkinter import (
    BOTH,
    END,
    HORIZONTAL,
    LEFT,
    SUNKEN,
    BooleanVar,
    Button,
    Checkbutton,
    Entry,
    Listbox,
    Menu,
    S,
    StringVar,
    Text,
    Tk,
    X,
    messagebox,
    simpledialog,
)
from tkinter.ttk import Frame, Label, PanedWindow, Treeview
from typing import Any, Optional


class FieldTypes(Enum):
    Array = auto()
    Object = auto()
    String = auto()
    Number = auto()
    Boolean = auto()


class SchemaField:
    def __init__(
        self,
        _id: str,
        path: list[str],
        field_type: FieldTypes,
        enums: Optional[list[str]] = None,
        description: Optional[str] = None,
    ):
        self.id = _id
        self.description = description
        self.path = path
        self.type = field_type
        self.enums = enums


class GHEDesignerWindow(Tk):
    """This form is the primary GUI entry point for the program; all control runs through here"""

    def __init__(self) -> None:
        super().__init__(className="GHEDesigner Inputs Assistant")

        # set some basic program information like title and an icon
        self.title("GHEDesignerWindow")

        # setup event listeners
        self._gui_queue: Queue[Any] = Queue()
        self._check_queue()

        # define the Tk.Variable instances that will be used to communicate with the GUI widgets
        self._define_tk_variables()

        # define a few of the dynamic widgets that will come and go as the user selects nodes
        self.listbox: Optional[Listbox] = None

        # get the current schema contents to build out the GUI
        this_script = Path(__file__).resolve()
        gui_dir = this_script.parent
        package_root = gui_dir.parent
        schema_file = package_root / "ghedesigner.schema.json"
        self.node_data: dict[str, SchemaField] = {}
        self.schema = loads(schema_file.read_text())

        # not really a list of strings, but eventually a list of whatever exists where we are in the schema
        self.data: list[str] = []

        # for now just create a dummy input file
        # we aren't packaging demos with the library, so we can't import a live example
        self.current_config = {
            "version": 1,
            "topology": [{"type": "ground-heat-exchanger", "name": "ghe1"}],
            "fluid": {"fluid_name": "WATER", "concentration_percent": 0, "temperature": 20},
            "ground-heat-exchanger": {
                "ghe1": {
                    "soil": {"conductivity": 2, "rho_cp": 2343493, "undisturbed_temp": 18.3},
                    "pipe": {"inner_diameter": 0.03404, "outer_diameter": 0.04216, "arrangement": "SINGLEUTUBE"},
                }
            },
            "simulation-control": {
                "thermal-sizing-run": True,
            },
        }

        # build out the form and specify a minimum size, which may not be uniform across platforms
        self._build_gui()
        self.update()
        self.minsize(self.winfo_width(), self.winfo_height())

        # set up some important member variables
        self._thread_running = False
        self._inputs_are_dirty = False

        # window setup operations
        self._update_status_bar("Program Initialized")
        self._refresh_gui_state()
        self.bind("<Key>", self._handle_button_pressed)

    def _handle_button_pressed(self, event):
        # relevant_modifiers
        # mod_shift = 0x1
        mod_control = 0x4
        # mod_alt = 0x20000
        if event.keysym == "e" and mod_control & event.state:
            pass

    def _check_queue(self):
        """Checks the GUI queue for actions and sets a timer to check again each time"""
        while True:
            # noinspection PyBroadException
            try:
                task = self._gui_queue.get(block=False)
                self.after_idle(task, [])
            except Exception:  # noqa: BLE001
                break
        # noinspection PyTypeChecker
        self.after(100, self._check_queue)

    def _define_tk_variables(self):
        """Creates and initializes all the Tk.Variable instances used in the GUI for two-way communication"""
        self._tk_var_status = StringVar(value="Program Initialized")

    def expand_schema_tree(self, tree: Treeview, item: str = ""):
        for child in tree.get_children(item):
            self.expand_schema_tree(tree, child)
        tree.item(item, open=True)

    def file_open(self):
        pass

    def file_save(self):
        pass

    def file_exit(self):
        self.destroy()

    def _build_gui(self):
        """Builds out the entire window GUI, calling workers as necessary"""
        # now build the top menubar, it's not part of the geometry, but a config parameter on the root Tk object
        menubar = Menu(self)
        file_menu = Menu(menubar, tearoff=False)
        file_menu.add_command(label="Open", command=self.file_open)
        file_menu.add_command(label="Save", command=self.file_save)
        file_menu.add_command(label="Exit", command=self.file_exit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)

        # Create the main PanedWindow
        paned_window = PanedWindow(self, orient=HORIZONTAL)
        paned_window.pack(fill=BOTH, expand=True)

        # Left Panel: Schema Treeview
        tree_frame = Frame(paned_window, padding=5)
        paned_window.add(tree_frame, weight=1)

        Label(tree_frame, text="Schema View", font=("Arial", 14)).pack(side="top")

        self.tree = Treeview(tree_frame, show="tree")
        self.tree.pack(fill=BOTH, expand=True)
        self.load_schema("", self.schema)
        self.tree.bind("<<TreeviewSelect>>", self.on_select)

        self.expand_schema_tree(self.tree)

        # Middle Panel: Editor
        editor_frame = Frame(paned_window, padding=10, relief="ridge")
        paned_window.add(editor_frame, weight=1)
        Label(editor_frame, text="Editing Section", font=("Arial", 14)).pack(side="top")
        self.editing_canvas_frame = Frame(editor_frame, padding=10, relief="ridge")
        self.editing_canvas_frame.pack(fill=BOTH, expand=True)

        # Right Panel: Read-only JSON Viewer
        json_frame = Frame(paned_window, padding=5)
        paned_window.add(json_frame, weight=1)
        Label(json_frame, text="Live JSON Input File", font=("Arial", 14)).pack(side="top")
        self.json_viewer = Text(json_frame, wrap="word")
        self.json_viewer.pack(fill=BOTH, expand=True)
        self.update_input_file()  # should eventually be updated after any change

        # build the status bar
        status_frame = Frame(self)
        Label(status_frame, relief=SUNKEN, anchor=S, textvariable=self._tk_var_status).pack(
            side=LEFT, fill=BOTH, expand=True
        )
        status_frame.pack(anchor=S, fill=X)

    def load_schema(self, parent, schema, key="Root"):
        node_id = self.tree.insert(parent, "end", text=key)

        if "type" in schema:
            enums = None
            description = None
            if "description" in schema:
                description = schema["description"]
            if schema["type"] == "array":
                field_type = FieldTypes.Array
            elif schema["type"] == "object":
                field_type = FieldTypes.Object
            elif schema["type"] == "string":
                field_type = FieldTypes.String
                if "enum" in schema:
                    enums = schema["enum"]
            elif schema["type"] == "number":
                field_type = FieldTypes.Number
            elif schema["type"] == "boolean":
                field_type = FieldTypes.Boolean
            else:
                raise Exception(f"Invalid schema type: {schema['type']}")
            self.node_data[node_id] = SchemaField(node_id, ["a"], field_type, enums=enums, description=description)

        if "properties" in schema:
            for prop, details in schema["properties"].items():
                self.load_schema(node_id, details, prop)

        if "items" in schema:
            self.load_schema(node_id, schema["items"], "Array Items")

        # if "required" in schema:
        #     self.tree.insert(node_id, "end", text=f"(Required: {', '.join(schema['required'])})", tags=("info",))
        #
        # if "dependentRequired" in schema:
        #     for key, dependencies in schema["dependentRequired"].items():
        #         dep_node = self.tree.insert(node_id, "end", text=f"{key} → Requires: {', '.join(dependencies)}",
        #                                     tags=("dependency",))

        self.tree.tag_configure("info", foreground="blue")
        self.tree.tag_configure("dependency", foreground="red")

    def update_input_file(self):
        text = dumps(self.current_config, indent=2)
        self.json_viewer.delete("1.0", "end")
        self.json_viewer.insert("1.0", text)

    def on_select(self, _):
        item_id = self.tree.selection()
        for widget in self.editing_canvas_frame.winfo_children():
            widget.destroy()
        if item_id:
            first_selected = item_id[0]
            node_data = self.node_data[first_selected]
            path_segments = []
            while item_id:
                path_segments.insert(0, self.tree.item(item_id, "text"))  # Insert at the beginning
                item_id = self.tree.parent(item_id)
            item_text = "/".join(path_segments)
            Label(self.editing_canvas_frame, text=item_text).pack()
            Label(self.editing_canvas_frame, text=node_data.type).pack()
            if node_data.description:
                Label(self.editing_canvas_frame, text=node_data.description).pack()
            if node_data.type == FieldTypes.String:
                entry_var = StringVar(value="<default>")
                # entry_var.trace("write", self.update_json)
                Entry(self.editing_canvas_frame, textvariable=entry_var).pack()
            if node_data.type == FieldTypes.Number:
                entry_var = StringVar(value="<default>")
                # entry_var.trace("write", self.update_json)
                Entry(self.editing_canvas_frame, textvariable=entry_var).pack()
            if node_data.type == FieldTypes.Boolean:
                entry_var = BooleanVar(value=True)
                # entry_var.trace("write", self.update_json)
                Checkbutton(self.editing_canvas_frame, variable=entry_var).pack()
            if node_data.type == FieldTypes.Array:
                self.listbox = Listbox(self.editing_canvas_frame, height=5)
                self.listbox.pack(fill="both", expand=True)
                self.update_listbox()
                buttons = Frame(self.editing_canvas_frame)
                buttons.pack(fill="x")
                Button(buttons, text="Add", command=self.add_item).pack(side="left", expand=True, fill="x")
                Button(buttons, text="Edit", command=self.edit_item).pack(side="left", expand=True, fill="x")
                Button(buttons, text="Remove", command=self.remove_item).pack(side="left", expand=True, fill="x")
        else:
            item_text = "*No Item Selected*"
            Label(self.editing_canvas_frame, text=item_text).pack()

    def update_listbox(self):
        """Refresh the listbox with the current array values."""
        self.listbox.delete(0, END)
        for item in self.data:
            self.listbox.insert(END, str(item))

    def add_item(self):
        """Prompt for a new item and add it to the list."""
        new_value = simpledialog.askstring("Add Item", "Enter new value:")
        if new_value is not None:
            self.data.append(new_value)
            self.update_listbox()

    def edit_item(self):
        """Edit the selected item in the listbox."""
        selected_index = self.listbox.curselection()
        if not selected_index:
            messagebox.showwarning("Edit Item", "Select an item to edit.")
            return

        index = selected_index[0]
        current_value = self.data[index]

        new_value = simpledialog.askstring("Edit Item", "Modify value:", initialvalue=current_value)
        if new_value is not None:
            self.data[index] = new_value
            self.update_listbox()

    def remove_item(self):
        """Remove the selected item from the list."""
        selected_index = self.listbox.curselection()
        if not selected_index:
            messagebox.showwarning("Remove Item", "Select an item to remove.")
            return

        index = selected_index[0]
        del self.data[index]
        self.update_listbox()

    def _refresh_gui_state(self):
        """Checks current instance flags and sets button states appropriately"""
        if self._thread_running:
            pass
        else:
            pass

    def _update_status_bar(self, extra_message: str) -> None:
        """
        Updates the status bar at the bottom of the window, providing data based on flags and displaying the message

        :param extra_message: String message to show on the right side of the status bar
        :return: Nothing
        """
        self._tk_var_status.set(extra_message)

    def run(self) -> None:
        """Executes the Tk main loop to handle all GUI events and update"""
        self.mainloop()
