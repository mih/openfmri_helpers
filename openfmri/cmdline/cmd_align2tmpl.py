# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the openfmri package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Align data to a template
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

import nipype.interfaces.io as nio           # Data i/o		
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
from nipype.interfaces.utility import Function
from ..nipype_helpers import *

def setup_parser(parser):
    hlp.parser_add_common_args(parser,
        opt=('datadir', 'dataset', 'subjects', 'workdir'))
    hlp.parser_add_common_args(parser, required=True,
        opt=('label',))
    parser.add_argument('--input-expression',
        help="""For the data input""")
    parser.add_argument('--template',
        help="""Reference template label""")
    parser.add_argument('--bet-padding', type=hlp.arg2bool,
        help="""Enable padding for BET""")
    parser.add_argument('--bet-frac', type=float,
        help="""Frac parameter for the skullstripping""")


def proc(label, tmpl_label, template, wf, subj, input, dsdir,
         bet_frac=0.5, bet_padding=False):
    import hashlib

    basename = os.path.basename(input)
    basename = basename[:basename.index('.')]
    brain_reference = opj(dsdir, 'templates', tmpl_label, 'brain.nii.gz')

    hash = hashlib.md5(input).hexdigest()
    sink = pe.Node(
            interface=nio.DataSink(
                parameterization=False,
                base_directory=os.path.abspath(os.path.dirname(input)),
                regexp_substitutions=[
                    ('/[^/]*\.nii', '_in_tmpl_%s.nii' % tmpl_label),
                ]),
            name="sub%.3i_sink_%s" % (subj, hash),
            overwrite=True)

    # mean vol for each input
    make_meanvol = pe.Node(
            name='sub%.3i_meanvol_%s' % (subj, hash),
            interface=fsl.ImageMaths(op_string='-Tmean'))
    make_meanvol.inputs.in_file = input
    # extract brain for later linear alignment
    if bet_padding:
        bet_interface=fsl.BET(padding=True)
    else:
        bet_interface=fsl.BET(robust=True)
    subj_bet = pe.Node(
            name='sub%.3i_meanbet_%s' % (subj, hash),
            interface=bet_interface)
    subj_bet.inputs.frac=bet_frac
    subj_bet.inputs.mask = True
    wf.connect(make_meanvol, 'out_file', subj_bet, 'in_file')

    #
    align2template = pe.Node(
            name='sub%.3i_align2tmpl_%s' % (subj, hash),
            interface=fsl.FLIRT(
                cost='corratio',
                no_search=True,
                uses_qform=True,
                #searchr_x=[-90, 90],
                #searchr_y=[-90, 90],
                #searchr_z=[-90, 90],
                dof=12,
                args="-interp trilinear"))
    wf.connect(subj_bet, 'out_file', align2template, 'in_file')
    wf.connect(template, 'out_file', align2template, 'reference')

    fix_xfm = pe.Node(
            name='sub%.3i_fixxfm_%s' % (subj, hash),
            interface=Function(
                function=fix_xfm_after_zpad,
                input_names=['xfm', 'reference', 'nslices'],
                output_names=['out_file']))
    #zpad_brain.inputs.nslices = zpad
    fix_xfm.inputs.nslices = 20
    wf.connect(template, 'out_file', fix_xfm, 'reference')
    wf.connect(align2template, 'out_matrix_file', fix_xfm, 'xfm')
    wf.connect(fix_xfm, 'out_file', sink, '%s_xfm.@out' % basename)


    mask2tmpl_lin = pe.Node(
            name='sub%.3i_mask2tmpl_lin_%s' % (subj, hash),
        interface=fsl.ApplyXfm(
            interp='nearestneighbour',
            reference=brain_reference,
            apply_xfm=True))
    wf.connect(fix_xfm, 'out_file', mask2tmpl_lin, 'in_matrix_file')
    wf.connect(subj_bet, 'mask_file', mask2tmpl_lin, 'in_file')
    wf.connect(mask2tmpl_lin, 'out_file', sink, '%s_brainmask.@out' % basename)

    return align2template, mask2tmpl_lin


def run(args):
    # label of the group template -- used to look up config options
    label = args.label
    cfg_section = 'align2tmpl %s' % label

    dataset = hlp.get_cfg_option('common', 'dataset', cli_input=args.dataset)

    subjects = hlp.get_dataset_subj_ids(args)
    subjects = hlp.exclude_subjects(subjects, cfg_section)

    dsdir = hlp.get_dataset_dir(args)

    wf_name = "align2tmpl_%s_%s" % (label, dataset)
    wf = hlp.get_base_workflow(wf_name.replace('.', '_'), args)

    input_exp = hlp.get_cfg_option(
                    cfg_section,
                    'input expression',
                    cli_input=args.input_expression)

    template = hlp.get_cfg_option(
                    cfg_section,
                    'template',
                    cli_input=args.template)

    bet_padding=hlp.arg2bool(
                hlp.get_cfg_option(
                    cfg_section,
                    'bet padding',
                    cli_input=args.bet_padding,
                    default=False))
    bet_frac=float(
                hlp.get_cfg_option(
                    cfg_section,
                    'bet frac',
                    cli_input=args.bet_frac,
                    default=0.5))

    brain_reference = opj(dsdir, 'templates', template, 'brain.nii.gz')
    zpad_brain = pe.Node(
            name='zpad_template_brain',
            interface=Function(
                function=zslice_pad,
                input_names=['in_file', 'nslices'],
                output_names=['out_file']))
    #zpad_brain.inputs.nslices = zpad
    zpad_brain.inputs.nslices = 20
    zpad_brain.inputs.in_file = brain_reference

    lin_masks_ = []
    lin_aligned_ = []
    for subj in subjects:
        subj_sink = pe.Node(
                interface=nio.DataSink(
                    parameterization=False,
                    base_directory=os.path.abspath(opj(dsdir, 'sub%.3i' % subj)),
                    regexp_substitutions=[
                        ('/[^/]*\.nii', '_in_tmpl_%s.nii' % template),
                    ]),
                name="sub%.3i_sink" % subj,
                overwrite=True)

        expr = input_exp % dict(subj='sub%.3i' % subj)

        df = nio.DataFinder(root_paths=dsdir, match_regex=expr,
                            ignore_exception=True)
        result = df.run()
        if result.outputs is None:
            continue

        subj_lin_aligned = pe.Node(
                name='sub%.3i_lin_aligned' % subj,
                interface=util.Merge(len(result.outputs.out_paths)))
        # do the same thing for each input file
        for i, input in enumerate(result.outputs.out_paths):
            lin_xfm, lin_mask = \
                proc(label, template, zpad_brain, wf, subj, input, dsdir, bet_frac=bet_frac,
                     bet_padding=bet_padding)
            lin_masks_.append(lin_mask)
            lin_aligned_.append(lin_xfm)
            wf.connect(lin_xfm, 'out_file', subj_lin_aligned, 'in%i' % (i + 1))

        # merge subj samples and store for QA
        subj_merge_lin_aligned = pe.Node(
                name='sub%.3i_merge_lin_aligned' % subj,
                interface=fsl.Merge(dimension='t'))
        wf.connect(subj_lin_aligned, 'out', subj_merge_lin_aligned, 'in_files')
        wf.connect(subj_merge_lin_aligned, 'merged_file',
                   subj_sink, 'qa.%s.lin_aligned_brain_samples.@out' % template)
    # store QA in template folder
    tmpl_sink = pe.Node(
            interface=nio.DataSink(
                parameterization=False,
                base_directory=os.path.abspath(opj(dsdir, 'templates', template)),
                regexp_substitutions=[
                    ('/[^/]*\.nii', '.nii'),
                ]),
            name="tmpl_sink",
            overwrite=True)
    # merge all masks across all subj and store for QA
    lin_masks = pe.Node(
            name='lin_masks',
            interface=util.Merge(len(lin_masks_)))
    lin_aligned = lin_masks.clone(name='lin_aligned')
    for i, mask in enumerate(lin_masks_):
        wf.connect(mask, 'out_file', lin_masks, 'in%i' % (i + 1))
    for i, aligned in enumerate(lin_aligned_):
        wf.connect(aligned, 'out_file', lin_aligned, 'in%i' % (i + 1))
    merge_lin_masks = pe.Node(
                name='merge_lin_masks',
                interface=fsl.Merge(dimension='t'))
    merge_lin_aligned = merge_lin_masks.clone(name='merge_lin_aligned')
    wf.connect(lin_masks, 'out', merge_lin_masks, 'in_files')
    wf.connect(lin_aligned, 'out', merge_lin_aligned, 'in_files')
    wf.connect(merge_lin_aligned, 'merged_file',
               tmpl_sink, 'qa.%s.lin_aligned_brain_samples.@out' % label)
    mask_stats = pe.Node(
            name='mask_stats',
            interface=fsl.ImageMaths(op_string='-Tmean'))
    wf.connect(merge_lin_masks, 'merged_file', mask_stats, 'in_file')
    wf.connect(mask_stats, 'out_file',
               tmpl_sink, 'qa.%s.brain_mask_stats.@out' % label)
    intersection_mask = pe.Node(
            name='mask_intersection',
            interface=fsl.ImageMaths(op_string='-thr 1'))
    wf.connect(mask_stats, 'out_file', intersection_mask, 'in_file')
    wf.connect(intersection_mask, 'out_file',
               tmpl_sink, 'qa.%s.brain_mask_intersection.@out' % label)

    return wf

