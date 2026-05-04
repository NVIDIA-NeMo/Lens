# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Sphinx configuration for nemo-lens documentation."""

import os
import sys

# -- Project information -----------------------------------------------------

project = "nemo-lens"
copyright = "2026, NVIDIA Corporation"
author = "NVIDIA Corporation"
try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("nemo-lens")
except PackageNotFoundError:
    release = "0.0.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]

# Autodoc2 is optional — can be skipped for faster iteration.
skip_autodoc = os.environ.get("SKIP_AUTODOC", "false").lower() == "true"
if not skip_autodoc:
    extensions.append("autodoc2")

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "superpowers/**"]

# -- MyST Parser -------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
    "attrs_block",
]
myst_heading_anchors = 5

# -- Autodoc2 ----------------------------------------------------------------

sys.path.insert(0, os.path.abspath("../src"))

if not skip_autodoc:
    autodoc2_packages = [
        {"path": "../src/nemo/lens", "module": "nemo.lens"},
    ]
    autodoc2_render_plugin = "myst"
    autodoc2_output_dir = "apidocs"

# -- HTML output -------------------------------------------------------------

html_theme = "nvidia_sphinx_theme"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NVIDIA/nemo-lens/",
            "icon": "fa-brands fa-github",
        }
    ],
}
