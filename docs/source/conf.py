# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

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
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.inheritance_diagram',
    'matplotlib.sphinxext.plot_directive',
    'sphinx_design',
    'myst_nb',
    # 'numpydoc'
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
