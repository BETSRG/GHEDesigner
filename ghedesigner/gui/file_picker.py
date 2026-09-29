"""Native path selection for the local GHEDesigner GUI service."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal, Protocol

PathKind = Literal["file", "directory"]


class PathChooser(Protocol):
    """Callable used by the GUI service to open a native path chooser."""

    def __call__(self, kind: PathKind, initial_path: str | None = None) -> str | None: ...


class PathChooserError(RuntimeError):
    """Raised when no supported native chooser can be opened."""


class DirectoryOpener(Protocol):
    """Callable used by the GUI service to reveal an output directory."""

    def __call__(self, directory: Path) -> None: ...


class DirectoryOpenError(RuntimeError):
    """Raised when no supported platform file browser can be opened."""


def _initial_directory(initial_path: str | None) -> str:
    if not initial_path:
        return str(Path.cwd())
    candidate = Path(initial_path).expanduser()
    if candidate.is_dir():
        return str(candidate.resolve())
    if candidate.parent.is_dir():
        return str(candidate.parent.resolve())
    return str(Path.cwd())


def _run_macos_chooser(kind: PathKind, initial_directory: str) -> str | None:
    executable = shutil.which("osascript")
    if executable is None:
        raise PathChooserError("macOS could not find the osascript path chooser.")
    script = """
on run argv
    set selectionKind to item 1 of argv
    set initialDirectory to item 2 of argv
    activate
    if selectionKind is "directory" then
        set selectedItem to choose folder with prompt "Choose an output directory" ¬
            default location (POSIX file initialDirectory)
    else
        set selectedItem to choose file with prompt "Choose a load data file" ¬
            default location (POSIX file initialDirectory)
    end if
    return POSIX path of selectedItem
end run
"""
    result = subprocess.run(  # noqa: S603 - executable is resolved locally and no shell is used.
        [executable, "-e", script, kind, initial_directory],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return result.stdout.rstrip("\r\n") or None
    if result.returncode == 1 and ("-128" in result.stderr or "User canceled" in result.stderr):
        return None
    raise PathChooserError(result.stderr.strip() or "macOS could not open the path chooser.")


def _run_zenity_chooser(kind: PathKind, initial_directory: str) -> str | None:
    executable = shutil.which("zenity")
    if executable is None:
        raise PathChooserError("Could not find the Zenity path chooser.")
    command = [executable, "--file-selection", f"--filename={initial_directory}/"]
    if kind == "directory":
        command.extend(["--directory", "--title=Choose an output directory"])
    else:
        command.append("--title=Choose a load data file")
    result = subprocess.run(  # noqa: S603 - executable is resolved locally and no shell is used.
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return result.stdout.rstrip("\r\n") or None
    if result.returncode == 1:
        return None
    raise PathChooserError(result.stderr.strip() or "Zenity could not open the path chooser.")


def _run_kdialog_chooser(kind: PathKind, initial_directory: str) -> str | None:
    executable = shutil.which("kdialog")
    if executable is None:
        raise PathChooserError("Could not find the KDialog path chooser.")
    option = "--getexistingdirectory" if kind == "directory" else "--getopenfilename"
    result = subprocess.run(  # noqa: S603 - executable is resolved locally and no shell is used.
        [executable, option, initial_directory],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return result.stdout.rstrip("\r\n") or None
    if result.returncode == 1:
        return None
    raise PathChooserError(result.stderr.strip() or "KDialog could not open the path chooser.")


_TK_CHOOSER_SCRIPT = """
import sys
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
kind, initial_directory = sys.argv[1:3]
if kind == "directory":
    selected = filedialog.askdirectory(parent=root, initialdir=initial_directory, mustexist=False)
else:
    selected = filedialog.askopenfilename(
        parent=root,
        initialdir=initial_directory,
        filetypes=(("CSV and text files", "*.csv *.txt"), ("All files", "*")),
    )
root.destroy()
print(selected)
"""


def _run_tk_chooser(kind: PathKind, initial_directory: str) -> str | None:
    result = subprocess.run(  # noqa: S603 - the current Python executable runs a fixed local script.
        [sys.executable, "-c", _TK_CHOOSER_SCRIPT, kind, initial_directory],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise PathChooserError(result.stderr.strip() or "Tk could not open the path chooser.")
    return result.stdout.rstrip("\r\n") or None


def choose_path(kind: PathKind, initial_path: str | None = None) -> str | None:
    """Open the platform's native chooser and return the selected absolute path."""
    initial_directory = _initial_directory(initial_path)
    if sys.platform == "darwin":
        return _run_macos_chooser(kind, initial_directory)
    if shutil.which("zenity"):
        return _run_zenity_chooser(kind, initial_directory)
    if shutil.which("kdialog"):
        return _run_kdialog_chooser(kind, initial_directory)
    return _run_tk_chooser(kind, initial_directory)


def open_directory(directory: Path) -> None:
    """Open an existing directory in the platform's native file browser."""
    if sys.platform == "darwin":
        executable = shutil.which("open")
        arguments = [str(directory)]
    elif sys.platform == "win32":
        executable = shutil.which("explorer")
        arguments = [str(directory)]
    elif executable := shutil.which("xdg-open"):
        arguments = [str(directory)]
    elif executable := shutil.which("gio"):
        arguments = ["open", str(directory)]
    else:
        raise DirectoryOpenError("No supported file browser command is available.")
    if executable is None:
        raise DirectoryOpenError("No supported file browser command is available.")
    command = [executable, *arguments]
    try:
        subprocess.Popen(  # noqa: S603 - executable is resolved locally and no shell is used.
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as error:
        raise DirectoryOpenError(f"Unable to open the output directory: {error}") from error
