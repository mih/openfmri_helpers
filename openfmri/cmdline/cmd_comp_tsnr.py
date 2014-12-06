# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the openfmri package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Compute temporal signal-to-noise ratio for 4D images
"""

__docformat__ = 'restructuredtext'

# magic line for manpage summary
# man: -*- % cache all files required to generate the portal

import argparse
import os
import logging

from os.path import join as opj

from openfmri import cfg
from . import helpers as hlp
import hashlib

lgr = logging.getLogger(__name__)
parser_args = dict(formatter_class=argparse.RawDescriptionHelpFormatter)


def setup_parser(parser):
    hlp.parser_add_common_args(parser,
        opt=('datadir', 'dataset', 'subjects', 'workdir'))
    hlp.parser_add_common_args(parser, required=False,
        opt=('label',))
    parser.add_argument('--input-expression',
        help="""For the data input""")

import sys
import os                                    # system functions
import nipype.interfaces.io as nio           # Data i/o		
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
from nipype.interfaces.utility import Function
from ..nipype_helpers import *

def run(args):
    # label of the group template -- used to look up config options
    label = args.label
    if label is None:
        label = 'default'
    cfg_section = 'comp_tsnr %s' % label

    dataset = hlp.get_cfg_option('common', 'dataset', cli_input=args.dataset)

    subjects = hlp.get_dataset_subj_ids(args)
    subjects = hlp.exclude_subjects(subjects, cfg_section)

    dsdir = hlp.get_dataset_dir(args)

    wf_name = "comp_tsnr_%s_%s" % (label, dataset)
    wf = hlp.get_base_workflow(wf_name.replace('.', '_'), args)

    input_exp = hlp.get_cfg_option(cfg_section, 'input expression',
                                   cli_input=args.input_expression)

    for subj in subjects:
        expr = input_exp % dict(subj='sub%.3i' % subj)

        df = nio.DataFinder(root_paths=dsdir, match_regex=expr,
                            ignore_exception=True)
        result = df.run()
        if result.outputs is None:
            # no data, nothing to do
            continue

        # do the same thing for each input file
        for i, input in enumerate(sorted(result.outputs.out_paths)):
            basename = os.path.basename(input)
            basename = basename[:basename.index('.')]
            hash = hashlib.md5(input).hexdigest()
            sink = pe.Node(
                    interface=nio.DataSink(
                        parameterization=False,
                        base_directory=os.path.abspath(os.path.dirname(input)),
                        regexp_substitutions=[
                            ('/[^/]*\.nii', '.nii'),
                        ]),
                    name="sub%.3i_%i_sink_%s" % (subj, i, hash),
                    overwrite=True)

            # compute tSNR
            tsnr = pe.Node(
                name='sub%.3i_tsnr_%s' % (subj, hash),
                interface=Function(
                    function=tsnr_from_filename,
                    input_names=['in_file'],
                    output_names=['out_file']))
            tsnr.inputs.in_file = input
            wf.connect(tsnr, 'out_file', sink, 'qa.%s_tsnr.@out' % basename)
        if expr == input_exp:
            # there was no subj placeholder, we can break
            break
    return wf

