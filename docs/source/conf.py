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

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

html_logo = '_static/logo.jpg'
html_favicon = '_static/favicon.ico'

html_theme_options = {
  "github_url": "https://github.com/ipu69/pyqumo",
  # "navbar_end": ["theme-switcher", "version-switcher", "navbar-icon-links"],
  # "switcher": {
  #     "version_match": version,
  # }
}
