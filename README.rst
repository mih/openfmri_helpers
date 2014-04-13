***********************************************************
Productivity helpers for working with openfmri.org datasets
***********************************************************

Few commands that help facilitate working with dataset in openfmri.org layout.
For now two commands are implemented:

mk_grptmpl:
  generate a subject group specific template image for T1w, T2w or
  EPI images with an iterative multi-level (non-)linear alignment
  procedure

align2tmpl:
  align data to a group template

All commands will dynamically generate Nipype workflows and support execution
on a single machine or in parallel (e.g. on a cluster). Tools from AFNI and FSL
are used underneath.

The project is designed to be easily extensible with more functionality and
contributions are welcome!

Documentation isn't a particular strength, but each command comes with a manpage.

.. link list

`Bug tracker <https://github.com/hanke/openfmri_helpers/issues>`_ |
`Build status <http://travis-ci.org/hanke/openfmri_helpers>`_ |
`Documentation <https://openfmri_helpers.readthedocs.org>`_ |
`Downloads <https://github.com/hanke/openfmri_helpers/tags>`_ |
`PyPi <http://pypi.python.org/pypi/openfmri_helpers>`_
