# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

project = 'decodanda'
copyright = '2023, Lorenzo Posani'
author = 'Lorenzo Posani'
release = '0.6.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx_rtd_theme',
              'sphinx.ext.napoleon',
              ]

napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True
add_module_names = False
html_theme = "sphinx_rtd_theme"

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
autoclass_content = 'both'

# Set the navigation depth to 2
html_theme_options = {
    'navigation_depth': 3,
    'collapse_navigation': False
}

# Set the default sidebar mode to "expanded"
html_sidebars = {
    '**': ['globaltoc.html', 'sourcelink.html', 'searchbox.html'],
    'auto': ['localtoc.html'],
}
# html_show_sourcelink = False
# html_show_sphinx = False
html_theme = 'sphinx_rtd_theme'

autodoc_default_options = {
    'member-order': 'groupwise'
}

# html_use_modindex = True
# html_domain_indices = True