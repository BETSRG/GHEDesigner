# Graphical Network Editor

GHEDesigner includes a draft graphical editor for assembling district thermal-network inputs. The editor is intentionally separate from the simulation engine: it saves an editable project graph and exports a GHEDesigner input JSON file.

## Run From A Checkout

Install the normal development environment, then launch the GUI module:

```bash
uv sync
uv run python -m ghedesigner.gui
```

If you are using an already active virtual environment with GHEDesigner dependencies installed, run:

```bash
python -m ghedesigner.gui
```

Linux systems may need the Tk runtime package installed by the operating system, for example `python3-tk` on Debian or Ubuntu.

## Run After Installation

When the package is installed from this checkout, the GUI command is:

```bash
ghedesigner-gui
```

## Basic Workflow

1. Add components from the left palette.
2. Drag components on the canvas to arrange the network.
3. Select a component, choose **Set selected as upstream**, then click its downstream component. The editor replaces any existing upstream or downstream link needed to keep the one-pipe topology valid.
4. Edit the selected component's JSON in the right panel and apply the change.
5. Select an output folder and choose **Execute simulation** to run the current network from the GUI. The GUI writes `ghedesigner_gui_input.json` into the selected folder before running.
6. Save the editable project as `*.ghed-network.json` or export a GHEDesigner input as `*.ghedesigner.json`.
7. Exported inputs can also be run with the existing CLI:

```bash
ghedesigner path/to/project.ghedesigner.json ./tmp
```

The first draft focuses on one-pipe ordered topology export. Branching or multiple upstream connections are reported as validation issues before export.
