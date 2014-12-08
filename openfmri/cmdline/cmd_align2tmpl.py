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
        opt=('datadir', 'dataset', 'subjects', 'workdir', 'input_expression', 'zslice_padding'))
    hlp.parser_add_common_args(parser, required=True,
        opt=('label',))
    parser.add_argument('--template',
        help="""Reference template label""")
    parser.add_argument('--bet-padding', type=hlp.arg2bool,
        help="""Enable padding for BET""")
    parser.add_argument('--bet-frac', type=float,
        help="""Frac parameter for the skullstripping""")
    parser.add_argument('--search-radius', type=int,
        help="""For linear alignment""")
    parser.add_argument('--warp-resolution', type=float,
        help="""For non-linear alignment""")
    parser.add_argument('--non-linear', type=hlp.arg2bool,
        help="""Perform an additional non-linear step for alignment""")
    parser.add_argument('--motion-correction', type=hlp.arg2bool,
        help="""Perform motion correction on input prior to alignment""")
    parser.add_argument('--use-qform', type=hlp.arg2bool,
        help="""Whether to pass the -usesqform flag to FLIRT""")
    parser.add_argument('--input-subjtmpl', type=hlp.arg2bool,
        help="""Whether to treat the input as a subject template's head image.
        This will cause the algorithm to re-use existing brain masks and output
        results into a template-like directory structure.""")


def proc(label, tmpl_label, template, wf, subj, input, dsdir,
         motion_correction,
         non_linear=False, bet_frac=0.5, bet_padding=False, search_radius=0,
         warp_resolution=5, zpad=0, use_qform=True, input_subjtmpl=False):
    import hashlib

    inputdir = os.path.dirname(input)
    basename = os.path.basename(input)
    basename = basename[:basename.index('.')]
    brain_reference = opj(dsdir, 'templates', tmpl_label, 'brain.nii.gz')
    head_reference = opj(dsdir, 'templates', tmpl_label, 'head.nii.gz')

    # where does the output go?
    if input_subjtmpl:
        sinkslot_data = 'in_%s.head.@out' % (tmpl_label,)
        sinkslot_brainmask = 'in_%s.brain_mask.@out' % (tmpl_label,)
        sinkslot_affine = 'in_%s.subj2tmpl_12dof.@out' % (tmpl_label,)
        sinkslot_warpfield = 'in_%s.subj2tmpl_warp.@out' % (tmpl_label,)
        sink_regexp_substitutions=[
                    ('/[^/]*\.par', '.txt'),
                    ('/[^/]*\.nii', '.nii'),
                    ('/[^/]*\.mat', '.mat')]
    else:
        sinkslot_brainmask = '%s_brainmask.@out' % basename
        sinkslot_affine = '%s_xfm.@out' % basename
        sinkslot_data = '%s.@out' % basename
        sinkslot_warpfield = '%s_warp.@out' % basename
        sink_regexp_substitutions=[
                    ('/[^/]*\.par', '_%s.txt' % label),
                    ('/[^/]*\.nii', '_%s.nii' % label),
                    ('/[^/]*\.mat', '_%s.mat' % label)]

    hash = hashlib.md5(input).hexdigest()
    sink = pe.Node(
            interface=nio.DataSink(
                parameterization=False,
                base_directory=os.path.abspath(os.path.dirname(input)),
                regexp_substitutions=sink_regexp_substitutions),
            name="sub%.3i_sink_%s" % (subj, hash),
            overwrite=True)

    if not input_subjtmpl:
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

    align2template = pe.Node(
            name='sub%.3i_align2tmpl_%s' % (subj, hash),
            interface=fsl.FLIRT(
                cost='corratio',
                uses_qform=use_qform,
                dof=12,
                args="-interp trilinear"))
    if search_radius:
        align2template.inputs.searchr_x = [(-1) * search_radius, search_radius]
        align2template.inputs.searchr_y = [(-1) * search_radius, search_radius]
        align2template.inputs.searchr_z = [(-1) * search_radius, search_radius]
    else:
        align2template.inputs.no_search = True
    wf.connect(template, 'out_file', align2template, 'reference')
    if input_subjtmpl:
        # construct the existing brain image from the input
        align2template.inputs.in_file = opj(inputdir, 'brain.nii.gz')
    else:
        wf.connect(subj_bet, 'out_file', align2template, 'in_file')

    fix_xfm = pe.Node(
            name='sub%.3i_fixxfm_%s' % (subj, hash),
            interface=Function(
                function=fix_xfm_after_zpad,
                input_names=['xfm', 'reference', 'nslices'],
                output_names=['out_file']))
    fix_xfm.inputs.nslices = zpad
    wf.connect(template, 'out_file', fix_xfm, 'reference')
    wf.connect(align2template, 'out_matrix_file', fix_xfm, 'xfm')

    mask2tmpl_lin = pe.Node(
            name='sub%.3i_mask2tmpl_lin_%s' % (subj, hash),
        interface=fsl.ApplyXfm(
            interp='nearestneighbour',
            reference=brain_reference,
            apply_xfm=True))
    wf.connect(fix_xfm, 'out_file', mask2tmpl_lin, 'in_matrix_file')
    if input_subjtmpl:
        # construct the existing brain mask image from the input
        mask2tmpl_lin.inputs.in_file = opj(inputdir, 'brain_mask.nii.gz')
    else:
        wf.connect(subj_bet, 'mask_file', mask2tmpl_lin, 'in_file')
    if not non_linear:
        wf.connect(mask2tmpl_lin, 'out_file', sink, sinkslot_brainmask)

    # case of motion correcting input prior to alignment
    if motion_correction:
        mcflirt = pe.Node(
            name='sub%.3i_mcflirt_%s' % (subj, hash),
            interface=fsl.MCFLIRT(
                save_plots=True,
                stages=4,
                in_file=input,
                interpolation='sinc'))
        # use mean brain as MC target
        wf.connect(make_meanvol, 'out_file', mcflirt, 'ref_file')
        # write out MC estimates
        wf.connect(mcflirt, 'par_file', sink, '%s_moco.@out' % basename)

    if not non_linear:
        data2tmpl_lin = pe.Node(
                name='sub%.3i_data2tmpl_lin_%s' % (subj, hash),
            interface=fsl.ApplyXfm(
                interp='trilinear',
                reference=brain_reference,
                apply_xfm=True))
        if motion_correction:
            wf.connect(mcflirt, 'out_file', data2tmpl_lin, 'in_file')
        else:
            data2tmpl_lin.inputs.in_file = input
        wf.connect(fix_xfm, 'out_file', data2tmpl_lin, 'in_matrix_file')
        wf.connect(fix_xfm, 'out_file', sink, sinkslot_affine)
        wf.connect(data2tmpl_lin, 'out_file', sink, sinkslot_data)
        return align2template, mask2tmpl_lin

    # and non-linear
    align2template_nl = pe.Node(
            name='sub%.3i_align2tmpl_nonlin_%s' % (subj, hash),
            interface=fsl.FNIRT(
                intensity_mapping_model='global_non_linear_with_bias',
                field_file=True,
                ref_file=head_reference,
                warp_resolution=tuple([warp_resolution] * 3)))
    if input_subjtmpl:
        align2template_nl.inputs.in_file = input
    else:
        wf.connect(make_meanvol, 'out_file',
                   align2template_nl, 'in_file')
    wf.connect(fix_xfm, 'out_file',
               align2template_nl, 'affine_file')
    #wf.connect([(align2template_nl, subjsink, [
    #    ('field_file', 'qa.task%.3i.subj2tasktmpl_warp.@out' % task),
    #    ('warped_file', 'qa.task%.3i.subj2tasktmpl_nonlin.@out' % task),
    #    ])])

    # nonlin mask warping to the template
    warpmask2template = pe.Node(
        name='sub%.3i_warpmask2tmpl_nonlin_%s' % (subj, hash),
        interface=fsl.ApplyWarp(
            ref_file=brain_reference,
            interp='nn'))
    wf.connect(align2template_nl, 'field_file', warpmask2template, 'field_file')
    wf.connect(warpmask2template, 'out_file', sink, sinkslot_brainmask)
    if input_subjtmpl:
        # construct the existing brain mask image from the input
        warpmask2template.inputs.in_file = opj(inputdir, 'brain_mask.nii.gz')
    else:
        wf.connect(subj_bet, 'mask_file', warpmask2template, 'in_file')

    # nonlin data warping to the template
    warp2template = pe.Node(
        name='sub%.3i_warp2tmpl_nonlin_%s' % (subj, hash),
        interface=fsl.ApplyWarp(
            ref_file=brain_reference,
            interp='trilinear'))
    if motion_correction:
        wf.connect(mcflirt, 'out_file', warp2template, 'in_file')
    else:
        warp2template.inputs.in_file = input
    wf.connect(align2template_nl, 'field_file', warp2template, 'field_file')
    wf.connect(align2template_nl, 'field_file', sink, sinkslot_warpfield)
    wf.connect(warp2template, 'out_file', sink, sinkslot_data)

    return align2template_nl, warpmask2template


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

    non_linear_flag = hlp.arg2bool(hlp.get_cfg_option(cfg_section, 'non-linear',
                                   cli_input=args.non_linear))
    input_exp = hlp.get_cfg_option(cfg_section, 'input expression',
                                   cli_input=args.input_expression)
    template = hlp.get_cfg_option(cfg_section, 'template',
                                  cli_input=args.template)

    bet_padding=hlp.arg2bool(hlp.get_cfg_option(cfg_section, 'bet padding',
                                                cli_input=args.bet_padding,
                                                default=False))
    bet_frac=float(hlp.get_cfg_option(cfg_section, 'bet frac',
                                      cli_input=args.bet_frac, default=0.5))

    search_radius=int(hlp.get_cfg_option(cfg_section, 'search radius',
                                         cli_input=args.search_radius,
                                         default=0))
    warp_resolution=int(hlp.get_cfg_option(cfg_section, 'warp resolution',
                                           cli_input=args.warp_resolution,
                                           default=5))
    motion_correction=hlp.arg2bool(hlp.get_cfg_option(cfg_section, 'motion correction',
                                           cli_input=args.motion_correction,
                                           default=False))

    use_qform=hlp.arg2bool(hlp.get_cfg_option(cfg_section, 'use qform',
                                              cli_input=args.use_qform,
                                              default=True))

    input_subjtmpl=hlp.arg2bool(
            hlp.get_cfg_option(cfg_section,
                               'input is subject template',
                               cli_input=args.input_subjtmpl,
                               default=False))

    zpad = int(hlp.get_cfg_option(cfg_section, 'zslice padding',
                                  cli_input=args.zslice_padding, default=0))

    brain_reference = opj(dsdir, 'templates', template, 'brain.nii.gz')
    zpad_brain = pe.Node(
            name='zpad_template_brain',
            interface=Function(
                function=zslice_pad,
                input_names=['in_file', 'nslices'],
                output_names=['out_file']))
    zpad_brain.inputs.nslices = zpad
    zpad_brain.inputs.in_file = brain_reference

    masks_ = []
    aligned_ = []
    for subj in subjects:
        subj_sink = pe.Node(
                interface=nio.DataSink(
                    parameterization=False,
                    base_directory=os.path.abspath(opj(dsdir, 'sub%.3i' % subj)),
                    regexp_substitutions=[
                        ('/[^/]*\.nii', '_aligned_%s.nii' % label),
                    ]),
                name="sub%.3i_sink" % subj,
                overwrite=True)

        expr = input_exp % dict(subj='sub%.3i' % subj)

        df = nio.DataFinder(root_paths=dsdir, match_regex=expr,
                            ignore_exception=True)
        result = df.run()
        if result.outputs is None:
            continue

        subj_aligned = pe.Node(
                name='sub%.3i_aligned' % subj,
                interface=util.Merge(len(result.outputs.out_paths)))
        # do the same thing for each input file
        for i, input in enumerate(sorted(result.outputs.out_paths)):
            xfm, mask = \
                proc(label, template, zpad_brain, wf, subj, input, dsdir,
                     motion_correction,
                     non_linear=non_linear_flag, bet_frac=bet_frac,
                     bet_padding=bet_padding, search_radius=search_radius,
                     warp_resolution=warp_resolution, zpad=zpad,
                     use_qform=use_qform, input_subjtmpl=input_subjtmpl)
            masks_.append(mask)
            aligned_.append(xfm)
            if non_linear_flag:
                wf.connect(xfm, 'warped_file', subj_aligned, 'in%i' % (i + 1))
            else:
                wf.connect(xfm, 'out_file', subj_aligned, 'in%i' % (i + 1))

        # merge subj samples and store for QA
        subj_merge_aligned = pe.Node(
                name='sub%.3i_merge_aligned' % subj,
                interface=fsl.Merge(dimension='t'))
        wf.connect(subj_aligned, 'out', subj_merge_aligned, 'in_files')
        wf.connect(subj_merge_aligned, 'merged_file',
                   subj_sink, 'qa.%s.aligned_brain_samples.@out' % template)
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
    masks = pe.Node(
            name='masks',
            interface=util.Merge(len(masks_)))
    aligned = masks.clone(name='aligned')
    for i, mask in enumerate(masks_):
        wf.connect(mask, 'out_file', masks, 'in%i' % (i + 1))
    for i, a in enumerate(aligned_):
        if non_linear_flag:
            wf.connect(a, 'warped_file', aligned, 'in%i' % (i + 1))
        else:
            wf.connect(a, 'out_file', aligned, 'in%i' % (i + 1))
    merge_masks = pe.Node(
                name='merge_masks',
                interface=fsl.Merge(dimension='t'))
    merge_aligned = merge_masks.clone(name='merge_aligned')
    wf.connect(masks, 'out', merge_masks, 'in_files')
    wf.connect(aligned, 'out', merge_aligned, 'in_files')
    wf.connect(merge_aligned, 'merged_file',
               tmpl_sink, 'qa.%s.aligned_brain_samples.@out' % label)
    mask_stats = pe.Node(
            name='mask_stats',
            interface=fsl.ImageMaths(op_string='-Tmean'))
    wf.connect(merge_masks, 'merged_file', mask_stats, 'in_file')
    wf.connect(mask_stats, 'out_file',
               tmpl_sink, 'qa.%s.brain_mask_stats.@out' % label)
    intersection_mask = pe.Node(
            name='mask_intersection',
            interface=fsl.ImageMaths(op_string='-thr 1'))
    wf.connect(mask_stats, 'out_file', intersection_mask, 'in_file')
    wf.connect(intersection_mask, 'out_file',
               tmpl_sink, 'qa.%s.brain_mask_intersection.@out' % label)

    return wf

