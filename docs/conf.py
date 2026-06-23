# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

project = "VeRL-Omni"
copyright = "2026 Bytedance Ltd. and/or its affiliates"
author = "VeRL-Omni contributors"

master_doc = "index"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

myst_heading_anchors = 3

napoleon_google_docstring = True
napoleon_numpy_docstring = False

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"

autosectionlabel_prefix_document = True

suppress_warnings = ["ref.duplicate", "ref.myst"]

autodoc_mock_imports = [
    "verl.utils.vllm",
    "verl.workers.rollout.vllm_rollout",
    "vllm_omni",
]
