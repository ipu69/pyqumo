# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import math
import matplotlib
import matplotlib.pyplot as plt

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyqumo'
copyright = '2023, IPU RAS, Lab.69'
author = 'Andrey Larionov'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# ,
# 'sphinx_design',
# ,
# ,

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.inheritance_diagram',
    'matplotlib.sphinxext.plot_directive',
    'sphinx_design',
    'myst_nb',
    'sphinx.ext.graphviz',
]

templates_path = ['_templates']
exclude_patterns = []
autosummary_generate = True
autodoc_typehints = 'signature'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = ['pyqumo.css']

html_logo = '_static/pyqumo_logo_slim.png'
html_favicon = '_static/favicon.ico'

html_theme_options = {
  "github_url": "https://github.com/ipu69/pyqumo",
  # "navbar_end": ["theme-switcher", "version-switcher", "navbar-icon-links"],
  # "switcher": {
  #     "version_match": version,
  # }
}

# ------------------------------
# Matplotlib settings
# ------------------------------
matplotlib.use('agg')
plt.ioff()

plot_include_source = True
plot_formats = [('png', 96)]
plot_html_show_formats = False
plot_html_show_source_link = False

phi = (math.sqrt(5) + 1)/2

font_size = 13*72/96.0  # 13 px

plot_rcparams = {
    'font.size': font_size,
    'axes.titlesize': font_size,
    'axes.labelsize': font_size,
    'xtick.labelsize': font_size,
    'ytick.labelsize': font_size,
    'legend.fontsize': font_size,
    'figure.figsize': (3*phi, 3),
    'figure.subplot.bottom': 0.2,
    'figure.subplot.left': 0.2,
    'figure.subplot.right': 0.9,
    'figure.subplot.top': 0.85,
    'figure.subplot.wspace': 0.4,
    'text.usetex': False,
}
