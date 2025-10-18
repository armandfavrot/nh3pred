import os
import sys

sys.path.insert(0, os.path.abspath('../../src'))  # chemin vers ton package

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'nh3pred'
copyright = '2025, Armand Favrot'
author = 'Armand Favrot'
release = "0.1.3"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # génère doc à partir des docstrings
    'sphinx.ext.napoleon',     # support Google/NumPy style docstrings
    'sphinx.ext.viewcode',     # ajoute des liens vers le code source
]

templates_path = ['_templates']
exclude_patterns = []

add_module_names = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
