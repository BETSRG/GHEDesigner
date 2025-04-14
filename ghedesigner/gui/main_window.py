from queue import Queue
from tkinter import (
    BOTH,
    EW,
    LEFT,
    NSEW,
    SUNKEN,
    TOP,
    Button,
    Frame,
    Label,
    Menu,
    S,
    StringVar,
    Tk,
    X,
)
from tkinter.ttk import LabelFrame  # ttk widgets
from typing import Any


class GHEDesignerWindow(Tk):
    """This form is the primary GUI entry point for the program; all control runs through here"""

    def __init__(self) -> None:
        """
        The main window of the parameter estimation tool GUI workflow.
        This window is an instance of a tk.Tk object
        """
        super().__init__(className="GHEDesignerWindow")

        # set some basic program information like title and an icon
        self.title("GHEDesignerWindow")

        # setup event listeners
        self._gui_queue: Queue[Any] = Queue()
        self._check_queue()

        # define the Tk.Variable instances that will be used to communicate with the GUI widgets
        self._define_tk_variables()

        # build out the form and specify a minimum size, which may not be uniform across platforms
        self._build_gui()
        self.update()
        self.minsize(self.winfo_width(), self.winfo_height())

        # set up some important member variables
        self._thread_running = False

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
            try:
                task = self._gui_queue.get(block=False)
                self.after_idle(task, [])
            except Exception:  # noqa: BLE001
                break
        self.after(100, self._check_queue)

    def _define_tk_variables(self):
        """Creates and initializes all the Tk.Variable instances used in the GUI for two-way communication"""
        self._tk_var_status = StringVar(value="Program Initialized")

    def _build_gui(self):
        """Builds out the entire window GUI, calling workers as necessary"""
        # now build the top menubar, it's not part of the geometry, but a config parameter on the root Tk object
        menubar = Menu(self)
        menu_exit = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Exit", menu=menu_exit)
        self.config(menu=menubar)

        # main contents
        label_frame_control = LabelFrame(self, text="Stuff")
        Label(label_frame_control, text="Here is where we could put some stuff").pack(side=TOP, padx=3, pady=3)
        self._button_engage = Button(label_frame_control, text="Button", command=self._button)
        self._button_engage.pack(side=TOP, padx=3, pady=3, fill=X)
        label_frame_control.grid(row=0, column=0, sticky=NSEW)

        # build the status bar
        status_frame = Frame(self)
        Label(status_frame, relief=SUNKEN, anchor=S, textvariable=self._tk_var_status).pack(
            side=LEFT, fill=BOTH, expand=True
        )
        status_frame.grid(row=1, column=0, columnspan=3, sticky=EW)

        # set up the weight of each row (even) and column (distributed) for a nice looking GUI
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def _button(self):
        pass

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
