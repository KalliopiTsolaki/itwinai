# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../use-cases/'))
sys.path.insert(0, os.path.abspath('../use-cases/3dgan/'))
sys.path.insert(0, os.path.abspath('../use-cases/mnist/torch-lightning/'))
sys.path.insert(0, os.path.abspath('../src/itwinai'))
sys.path.insert(0, os.path.abspath('../'))

project = 'itwinai'
copyright = '2024, Matteo Bunino, Alexander Zoechbauer, Kalliopi Tsolaki'
author = 'Matteo Bunino, Alexander Zoechbauer, Kalliopi Tsolaki'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest', 'sphinx.ext.viewcode']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme' #'alabaster'
html_static_path = ['_static']
