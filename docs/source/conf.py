# Configuration file for the Sphinx documentation builder.

import os
import sys

# Path setup
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

project = "ocd-gd"
copyright = "2026, Asad Arshad"
author = "Asad Arshad"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autodoc_mock_imports = ["agama"]

# -- Napoleon configuration (Fixes NumPy docstrings) -----------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_admonition_for_examples = True  # Formats Examples into clean blocks
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# -- Autodoc configuration --------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"  # Or "sphinx_rtd_theme" / "alabaster"
html_static_path = ["_static"]
