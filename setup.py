# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the openfmri_helper package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""""""

import os
import sys
from os.path import join as opj
import openfmri
from distutils.core import setup
from glob import glob


__docformat__ = 'restructuredtext'

extra_setuptools_args = {}

def main(**extra_args):
    setup(name         = 'openfmri_helpers',
          version      = openfmri.__version__,
          author       = 'Michael Hanke and the openfmri_helpers developers',
          author_email = 'michael.hanke@gmail.com',
          license      = 'MIT License',
          url          = 'https://github.com/hanke/openfmri_helpers',
          download_url = 'https://github.com/hanke/openfmri_helpers/tags',
          description  = 'Misc helpers for (f)MRI datasets in openfmri layout',
          long_description = open('README.rst').read(),
          classifiers  = ["Development Status :: 3 - Alpha",
                          "Environment :: Console",
                          "Intended Audience :: Science/Research",
                          "License :: OSI Approved :: MIT License",
                          "Operating System :: OS Independent",
                          "Programming Language :: Python",
                          "Topic :: Scientific/Engineering"],
          platforms    = "OS Independent",
          provides     = ['openfmri'],
          # please maintain alphanumeric order
          packages     = [ 'openfmri',
                           'openfmri.cmdline',
                           ],
          scripts      = glob(os.path.join('bin', '*'))
          )

if __name__ == "__main__":
    main(**extra_setuptools_args)
