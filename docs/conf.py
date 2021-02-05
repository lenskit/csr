import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import sphinx_rtd_theme

import csr

project = 'CSR'
copyright = '2021 Boise State University'
author = 'Michael D. Ekstrand'

release = csr.__version__

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme'
]

source_suffix = '.rst'

pygments_style = 'sphinx'
highlight_language = 'python3'

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'github_user': 'lenskit',
    'github_repo': 'csr',
    'travis_button': False,
    'canonical_url': 'https://csr.lenskit.org/'
}
templates_path = ['_templates']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None)
}

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource'
}
