# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the openfmri package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Align images to a subject template
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
    hlp.parser_add_common_args(parser, required=True,
        opt=('label',))
    parser.add_argument('--template',
        help="""Reference template label""")
    parser.add_argument('--input-expression',
        help="""For the data input""")
    parser.add_argument('--fgthresh', type=float,
        help="""Threshold for foreground voxels in percent of the
        robust range. This threshold is used to determine a foreground
        mask for each aligned volume. The minimum projections through
        time of these masks are available as QA output. This parameter
        has no influence on the actual alignment.""")
    parser.add_argument('--via',
        help="""Label of a subject template to look up a transformed
        reference image in. This can be used of direct alignment is too
        unstable.""")

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
    cfg_section = 'align2subjtmpl %s' % label

    dataset = hlp.get_cfg_option('common', 'dataset', cli_input=args.dataset)

    subjects = hlp.get_dataset_subj_ids(args)
    subjects = hlp.exclude_subjects(subjects, cfg_section)

    dsdir = hlp.get_dataset_dir(args)

    wf_name = "align2subjtmpl_%s_%s" % (label, dataset)
    wf = hlp.get_base_workflow(wf_name.replace('.', '_'), args)

    input_exp = hlp.get_cfg_option(cfg_section, 'input expression',
                                   cli_input=args.input_expression)
    template = hlp.get_cfg_option(cfg_section, 'template',
                                  cli_input=args.template)
    via = hlp.get_cfg_option(cfg_section, 'via template',
                             cli_input=args.via)
    fgthresh = float(hlp.get_cfg_option(cfg_section, 'foreground threshold',
                                      cli_input=args.fgthresh, default=5.0))

    for subj in subjects:
        subj_sink = pe.Node(
                interface=nio.DataSink(
                    parameterization=False,
                    base_directory=os.path.abspath(opj(dsdir, 'sub%.3i' % subj,
                                                   'templates')),
                    container=template,
                    regexp_substitutions=[
                        ('/[^/]*\.nii', '_%s.nii' % label),
                    ]),
                name="sub%.3i_sink" % subj,
                overwrite=True)

        reference = opj(dsdir, 'sub%.3i' % subj,
                        'templates', template, 'head.nii.gz')
        tmpl_brain_mask = opj(dsdir, 'sub%.3i' % subj,
                        'templates', template, 'brain_mask.nii.gz')
        if not os.path.exists(tmpl_brain_mask):
            tmpl_brain_mask = None

        expr = input_exp % dict(subj='sub%.3i' % subj)

        df = nio.DataFinder(root_paths=dsdir, match_regex=expr,
                            ignore_exception=True)
        result = df.run()
        if result.outputs is None:
            # no data, nothing to do
            continue

        subj_aligned = pe.Node(
                name='sub%.3i_aligned' % subj,
                interface=util.Merge(len(result.outputs.out_paths)))
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
                            ('/[^/]*\.par', '_%s.txt' % label),
                            ('/[^/]*\.nii', '_%s.nii' % label),
                            ('/[^/]*\.mat', '_%s.mat' % label),
                        ]),
                    name="sub%.3i_%i_sink_%s" % (subj, i, hash),
                    overwrite=True)

            # 4D alignment to template
            align4d = pe.Node(
                    name='sub%.3i_%i_align4d_%s' % (subj, i, hash),
                    interface=fsl.MCFLIRT(
                        cost='corratio',
                        #cost='normmi',
                        save_plots=True,
                        stages=4))
            align4d.inputs.in_file = input
            wf.connect(align4d, 'out_file', sink, '%s.@out' % basename)
            wf.connect(align4d, 'par_file', sink, 'qa.%s_moest.@out' % basename)
            # look up an initial xfm to constrain the search
            if not via is None:
                reference = opj(dsdir, 'sub%.3i' % subj, 'templates',
                                 via, 'in_%s' % template, 'brain.nii.gz')
                if not os.path.exists(reference):
                    raise ValueError('"via template" does not exist')
            align4d.inputs.ref_file = reference
            # a bit of QA: compute a mask indicating voxels that are
            # non-background for the entire during of the timeseries
            # non-background is defined as less the X% of the robust range
            foregroundmask = pe.Node(
                name='sub%.3i_%i_foregroundmask_%s' % (subj, i, hash),
                interface=fsl.Threshold(
                    thresh=fgthresh,
                    args=' -bin -Tmin',
                    use_robust_range=True,
                    output_datatype='char'))
            wf.connect(align4d, 'out_file', foregroundmask, 'in_file')
            wf.connect(foregroundmask, 'out_file',
                       sink, 'qa.%s_fgmask.@out' % basename)
            wf.connect(foregroundmask, 'out_file',
                       subj_aligned, 'in%i' % (i + 1))
        # merge subj fgmasks and store for QA
        subj_merge_aligned = pe.Node(
                name='sub%.3i_merge_aligned' % subj,
                interface=fsl.Merge(dimension='t'))
        wf.connect(subj_aligned, 'out', subj_merge_aligned, 'in_files')
        wf.connect(subj_merge_aligned, 'merged_file',
                   subj_sink, 'qa.fgmasks.@out')
        jointfgmask = pe.Node(
            name='sub%.3i_jointfgmask' % (subj,),
            interface=fsl.maths.MathsCommand(
                args=' -Tmin -bin ',
                output_datatype='char'))
        wf.connect(subj_merge_aligned, 'merged_file',
                   jointfgmask, 'in_file')
        wf.connect(jointfgmask, 'out_file',
                   subj_sink, 'qa.jointfgmask.@out')
        if not tmpl_brain_mask is None:
            jointfgbrainmask = pe.Node(
                name='sub%.3i_jointfgbrainmask' % (subj,),
                interface=fsl.maths.MathsCommand(
                    args=' -mul "%s" ' % tmpl_brain_mask,
                    output_datatype='char'))
            wf.connect(jointfgmask, 'out_file',
                       jointfgbrainmask, 'in_file')
            wf.connect(jointfgbrainmask, 'out_file',
                       subj_sink, 'qa.jointfgbrainmask.@out')
    return wf

