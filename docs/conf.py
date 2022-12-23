# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from pathlib import Path
from sys import path

root_dir = Path(__file__).parent.parent.resolve()
path.insert(0, str(root_dir))

import datetime

from ghedesigner import VERSION


def get_author_names():
    with open(Path(__file__).parent.parent / "contributors.txt") as f:
        lines = f.readlines()

    # get author names
    full_names = [x.split("<")[0].strip() for x in lines]
    auth_names = ", ".join(full_names)

    # get copyright names
    last_names = [x.split(" ")[-1] for x in full_names]
    first_initials = [x.split(" ")[0][0] for x in full_names]
    cr_names = ", ".join([f"{x}, {y}." for x, y in zip(last_names, first_initials)])

    return auth_names, cr_names


author_names, copyright_names = get_author_names()

project = 'GHEDesigner'
copyright = f'{datetime.date.today().year}, {copyright_names}'
author = author_names
release = VERSION

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.viewcode', 'sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
