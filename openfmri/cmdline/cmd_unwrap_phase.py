# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the openfmri package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unwrap phase images.

At present, only magnitude/phase image pairs can be processed. Unwrapped phase
images are placed into the same directory as the raw phase input images, with
a file name suffix '_unwrapped'.
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

lgr = logging.getLogger(__name__)
parser_args = dict(formatter_class=argparse.RawDescriptionHelpFormatter)

def setup_parser(parser):
    hlp.parser_add_common_args(parser, required=True,
        opt=('label',))
    hlp.parser_add_common_opt(parser, 'input_expression',
            names=('--magnitude-img-expression',))
    hlp.parser_add_common_opt(parser, 'input_expression',
            names=('--phase-img-expression',))
    hlp.parser_add_common_args(parser,
        opt=('datadir', 'dataset', 'subjects', 'workdir'))

import sys
import os                                    # system functions
import nipype.interfaces.io as nio           # Data i/o		
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl
from ..nipype_helpers import *

def run(args):
    # label of the group template -- used to look up config options
    label = args.label
    cfg_section = 'unwrap phase %s' % label

    dataset = hlp.get_cfg_option('common', 'dataset', cli_input=args.dataset)

    subjects = hlp.get_dataset_subj_ids(args)
    subjects = hlp.exclude_subjects(subjects, cfg_section)

    dsdir = hlp.get_dataset_dir(args)

    wf_name = "unwrap_phase_%s_%s" % (label, dataset)
    wf = hlp.get_base_workflow(wf_name.replace('.', '_'), args)

    mag_exp = hlp.get_cfg_option(cfg_section, 'magnitude img expression',
                                 cli_input=args.magnitude_img_expression)
    pha_exp = hlp.get_cfg_option(cfg_section, 'phase img expression',
                                 cli_input=args.phase_img_expression)

    filesrcs = \
        dict([(subj,
               [hlp.get_data_finder(
                    'sub%.3i_%ssrc' % (subj, itype),
                    dsdir,
                    expr % dict(subj='sub%.3i' % subj))
                        for itype, expr in (('mag', mag_exp), ('pha', pha_exp))]
               ) for subj in subjects])
    wf = get_unwrap_workflow(wf, filesrcs, subjects, label, dsdir)

    return wf


def get_unwrap_workflow(wf, datasrcs, subjects, label, datadir):
    sinks = {}
    for subj in subjects:
        magsrcs = datasrcs[subj][0].run()
        phasrcs = datasrcs[subj][1].run()
        srcs = zip(magsrcs.outputs.out_paths, phasrcs.outputs.out_paths)
        for i, files in enumerate(srcs):
            # want to put each file into the folder the phase image came
            # from -- no other idea than to use a sink per file
            magfile, phafile = files
            sink_dir = os.path.dirname(phafile[len(datadir) + 1:])
            if not sink_dir in sinks:
                file_sink = pe.Node(
                    interface=nio.DataSink(
                        parameterization=False,
                        base_directory=os.path.dirname(phafile),
                        regexp_substitutions=[
                            ('/fid\d*', ''),
                        ]),
                    name="sub%.3i_sink%.3i" % (subj, i),
                    overwrite=True)
                sinks[sink_dir] = file_sink
            sink = sinks[sink_dir]
            prelude = pe.Node(
                interface=fsl.PRELUDE(
                    magnitude_file=magfile,
                    phase_file=phafile,
                    #process2d=True,
                    ),
                name="sub%.3i_prelude%.3i" % (subj, i))
            wf.connect(prelude, 'unwrapped_phase_file', sink, 'fid%i' % (i,))
    return wf
