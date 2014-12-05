# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the openfmri package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Create a subject template image
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
    hlp.parser_add_common_args(parser,
        opt=('datadir', 'dataset', 'subjects', 'workdir', 'zslice_padding'))
    hlp.parser_add_common_args(parser, required=True,
        opt=('label',))
    parser.add_argument(
        '--linear', type=int,
        help="""Number of iterations of linear alignments""")
    parser.add_argument(
        '--target-resolution', type=float, nargs=3, metavar='DIM',
        help="""Spatial target resolution of the subject template in XYZ.
        This should be given in the same unit as the pixel dimensions
        of the imnput images. Default: keep resolution.""")
    parser.add_argument(
        '--trim-roi', type=int, nargs=6, metavar='INT',
        help="""ROI to trim the subject space template to, given as
        x_min, x_size, y_min, y_size, z_min, z_size (voxel counts).
        Default: no trim""")
    parser.add_argument('--input-expression',
        help="""For the data input""")
    parser.add_argument('--bet-padding', type=hlp.arg2bool,
        help="""Enable padding for BET""")
    parser.add_argument('--tmpl-bet-frac', type=float,
        help="""Frac parameter for the template skullstripping""")
    parser.add_argument('--tmpl-bet-gradient', type=float,
        help="""Gradient parameter for the template skullstripping""")
    parser.add_argument('--initial-reference-brain', metavar='FILE',
        help="""Skull-stripped brain volume to use as an initial alignment.
        If not specified, one of the input images is used as a reference
        (the one with the smallest absolute difference from the mean of all
        input images).""")
    parser.add_argument('--use-4d-mean', type=hlp.arg2bool,
        help="""If enabled, 4D input images are "compressed" into their
        mean volume, which is then used for alignment. Otherwise the first
        volume is used.""")
    parser.add_argument('--fgthresh', type=float,
        help="""Threshold for foreground voxels in percent of the
        robust range. This threshold is used to determine a foreground
        mask for each aligned volume. The average of these foreground masks
        across all input samples is available as QA output. This parameter
        has no influence on the actual alignment.""")

import sys
import os                                    # system functions
import nipype.interfaces.io as nio           # Data i/o		
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
import nipype.interfaces.afni as afni
from nipype.interfaces.utility import Function
from ..nipype_helpers import *

def make_init_template(wf, subj, datasink, heads, target_resolution, zpad=0):
    # pick one subject as the 1st-level reference
    best_reference = pe.Node(
            name='sub%.3i_init_pick_best_reference' % (subj,),
            interface=Function(
                function=pick_closest_to_avg,
                input_names=['in_files', 'in_aux'],
                output_names=['out_head', 'none']))
    best_reference.inputs.in_aux = []
    wf.connect(heads, 'out_file', best_reference, 'in_files')

    init_zpad = pe.Node(
            name='sub%.3i_init_zpad' % (subj,),
            interface=Function(
                function=zslice_pad,
                input_names=['in_file', 'nslices'],
                output_names=['padded_file']))
    init_zpad.inputs.nslices = zpad
    if zpad > 0:
        wf.connect(best_reference, 'out_head', init_zpad, 'in_file')

    if not target_resolution is None:
        upsample = pe.Node(
                name='sub%.3i_init_upsample_init' % (subj,),
                interface=afni.Resample(
                    args='-rmode Cu -dxyz %s' % ' '.join(['%.2f' % i for i in target_resolution])))
        upsample.inputs.out_file='upsampled.nii.gz'
        upsample.inputs.outputtype='NIFTI_GZ'

        if zpad > 0:
            wf.connect(init_zpad, 'padded_file', upsample, 'in_file')
        else:
            wf.connect(best_reference, 'out_head', upsample, 'in_file')
        wf.connect(upsample, 'out_file', datasink, 'qa.init.head.@out')
        return upsample
    else:
        # no upsampling
        if zpad > 0:
            wf.connect(init_zpad, 'padded_file', datasink, 'qa.init.head.@out')
            return init_zpad
        else:
            wf.connect(best_reference, 'out_head', datasink, 'qa.init.head.@out')
            return best_reference

def make_subj_preproc_branch(wf, subj, datasrc, use_4d_mean=False):
    # normalize all input images
    normalize_vols = pe.MapNode(
            name='sub%.3i_normalize_samplevols' % (subj,),
            interface=fsl.ImageMaths(op_string='-inm 1000'),
            iterfield=['in_file'])
    if use_4d_mean:
        make_samplevols = pe.MapNode(
                name='sub%.3i_make_samplevol' % (subj,),
                interface=fsl.ImageMaths(op_string='-Tmean'),
                iterfield=['in_file'])
        wf.connect(make_samplevols, 'out_file', normalize_vols, 'in_file')
    else:
        make_samplevols = pe.MapNode(
                name='sub%.3i_make_samplevol' % (subj,),
                interface=fsl.ExtractROI(t_min=0, t_size=1),
                iterfield=['in_file'])
        wf.connect(make_samplevols, 'roi_file', normalize_vols, 'in_file')
    wf.connect(datasrc, 'out_paths', make_samplevols, 'in_file')
    return normalize_vols

def make_subj_lvl_branch(label, wf, subj, lvl, tmpl, vols, sink, fgthresh):
    if hasattr(tmpl.outputs, 'out_file'):
        oslot = 'out_file'
    else:
        oslot = 'out_head'

     # merge all samplevols to 4D for alignment
    merge_samplevols = pe.Node(
            name='sub%.3i_lvl%i_merge_samplevols' % (subj, lvl),
            interface=fsl.Merge(dimension='t'))
    wf.connect(vols, 'out_file', merge_samplevols, 'in_files')
    # 4D alignment across runs
    align_samplevols = pe.Node(
            name='sub%.3i_lvl%i_align_samplevols' % (subj, lvl),
            interface=fsl.MCFLIRT(
                interpolation='sinc',
                stages=4))
    wf.connect(tmpl, oslot, align_samplevols, 'ref_file')
    wf.connect(merge_samplevols, 'merged_file', align_samplevols, 'in_file')
    # MCFlirt introduced strange values on the edges if blank space or padded
    # slices, to aggressive threshold of the result
    zeroagain = pe.Node(
            name='sub%.3i_lvl%i_threshold_samplevols' % (subj, lvl),
            interface=fsl.Threshold(
                thresh=fgthresh,
                use_nonzero_voxels=False,
                use_robust_range=True))
    wf.connect(align_samplevols, 'out_file', zeroagain, 'in_file')
    wf.connect(zeroagain, 'out_file',
               sink, 'qa.lvl%i.aligned_head_samples.@out' % (lvl,))
    # merge all aligned volumes into a normalized subject template
    make_subj_tmpl = pe.Node(
            name='sub%.3i_lvl%i_make_subj_tmpl' % (subj, lvl),
            interface=Function(
                function=nonzero_avg,
                input_names=['in_file'],
                output_names=['out_file', 'avg_stats']))
    wf.connect(zeroagain, 'out_file', make_subj_tmpl, 'in_file')
    wf.connect(make_subj_tmpl, 'out_file',
               sink, 'qa.lvl%i.head.@out' % (lvl,))
    wf.connect(make_subj_tmpl, 'avg_stats',
               sink, 'qa.lvl%i.head_avgstats.@out' % (lvl,))
    return make_subj_tmpl

def trim_tmpl(wf, subj, tmpl, name, roi):
    # trim the template
    x_min, x_size, y_min, y_size, z_min, z_size = roi
    trim_template = pe.Node(
            name='sub%.3i_trim_%s' % (subj, name),
            interface=fsl.ExtractROI(
                x_min=x_min, x_size=x_size,
                y_min=y_min, y_size=y_size,
                z_min=z_min, z_size=z_size))
    wf.connect(tmpl, 'out_file', trim_template, 'in_file')
    return trim_template

def final_bet(wf, subj, head_tmpl, slot_name, padding=False, frac=0.5,
              gradient=0):
    if padding:
        interface=fsl.BET(mask=True, padding=True, frac=frac, vertical_gradient=gradient)
    else:
        interface=fsl.BET(mask=True, robust=True, frac=frac, vertical_gradient=gradient)
    bet = pe.Node(name='sub%.3i_final_bet' % (subj,), interface=interface)
    wf.connect(head_tmpl, slot_name, bet, 'in_file')
    return bet


def get_epi_tmpl_workflow(wf, datasrc, datasink,
                          subj,
                          label,
                          target_resolution,
                          datadir=os.path.abspath(os.curdir),
                          lin=3,
                          zpad=0,
                          template_roi=None,
                          bet_padding=False,
                          tmpl_bet_frac=0.5,
                          tmpl_bet_gradient=0,
                          init_template=None,
                          use_4d_mean=False,
                          fgthresh=5.0,
                          ):
    # extract sample volumes and normalize them
    vols = make_subj_preproc_branch(wf, subj, datasrc, use_4d_mean=use_4d_mean)

    tmpl = make_init_template(wf, subj, datasink, vols, target_resolution, zpad=zpad)

    for lvl in range(lin + 1):
        tmpl = make_subj_lvl_branch(label, wf, subj, lvl, tmpl, vols, datasink, fgthresh)

    if not template_roi is None:
        tmpl = trim_tmpl(wf, subj, tmpl, 'tmpl', roi=template_roi)
        out_slot = 'roi_file'
    else:
        out_slot = 'out_file'
    bet_tmpl = final_bet(wf, subj, tmpl, out_slot,
                     bet_padding, tmpl_bet_frac,
                     tmpl_bet_gradient)
    wf.connect(bet_tmpl, 'out_file', datasink, 'brain.@out')
    wf.connect(bet_tmpl, 'mask_file', datasink, 'brain_mask.@out')
    wf.connect(tmpl, out_slot, datasink, 'head.@out')

    return wf


def run(args):
    # label of the subject template -- used to look up config options
    label = args.label
    cfg_section = 'subject template %s' % label

    dataset = hlp.get_cfg_option('common', 'dataset', cli_input=args.dataset)

    target_res = hlp.get_cfg_option(cfg_section, 'resolution',
                                    cli_input=args.target_resolution)
    if isinstance(target_res, basestring):
        target_res = [float(i) for i in target_res.split()]

    roi = hlp.get_cfg_option(cfg_section, 'trim roi',
                        cli_input=args.trim_roi)
    if isinstance(roi, basestring):
        roi = [int(i) for i in roi.split()]

    subjects = hlp.get_dataset_subj_ids(args)
    subjects = hlp.exclude_subjects(subjects, cfg_section)

    dsdir = hlp.get_dataset_dir(args)

    wf_name = "subjtmpl_%s_%s" % (label, dataset)
    wf = hlp.get_base_workflow(wf_name.replace('.', '_'), args)

    input_exp = hlp.get_cfg_option(
                    cfg_section,
                    'input expression',
                    cli_input=args.input_expression)
    datasrcs = dict([(subj,
                      hlp.get_data_finder(
                        'sub%.3i_datasrc' % subj,
                        dsdir,
                        input_exp % dict(subj='sub%.3i' % subj)))
                            for subj in subjects])
    datasinks = dict([
        (subj,
         pe.Node(
            interface=nio.DataSink(
            parameterization=False,
            #TODO move to subj dir
            base_directory=os.path.abspath(opj(dsdir,
                                               'sub%.3i' % subj,
                                               'templates')),
            container=label,
            regexp_substitutions=[
                ('/[^/]*\.nii', '.nii'),
            ]),
            name="subj%.3i_datasink" % subj,
            overwrite=True)) for subj in subjects])

    for subj in subjects:
        wf = get_epi_tmpl_workflow(wf, datasrcs[subj], datasinks[subj], subj,
                label,
                target_res,
                dsdir,
                lin=int(hlp.get_cfg_option(cfg_section,
                                           'linear iterations',
                                           cli_input=args.linear,
                                           default=1)),
                zpad=int(hlp.get_cfg_option(cfg_section,
                                            'zslice padding',
                                            cli_input=args.zslice_padding,
                                            default=0)),
                template_roi=roi,
                bet_padding=hlp.arg2bool(
                            hlp.get_cfg_option(
                                cfg_section,
                                'bet padding',
                                cli_input=args.bet_padding,
                                default=False)),
                tmpl_bet_frac=float(
                            hlp.get_cfg_option(
                                cfg_section,
                                'template bet frac',
                                cli_input=args.tmpl_bet_frac,
                                default=0.5)),
                tmpl_bet_gradient=float(
                            hlp.get_cfg_option(
                                cfg_section,
                                'template bet gradient',
                                cli_input=args.tmpl_bet_gradient,
                                default=0)),
                init_template=hlp.get_cfg_option(
                                cfg_section,
                                'initial reference brain',
                                cli_input=args.initial_reference_brain,
                                default=None),
                use_4d_mean=hlp.arg2bool(
                            hlp.get_cfg_option(
                                cfg_section,
                                'use 4d mean',
                                cli_input=args.use_4d_mean,
                                default=False)),
                fgthresh = float(
                            hlp.get_cfg_option(
                                cfg_section,
                                'foreground threshold',
                                cli_input=args.fgthresh,
                                default=5.0)),
                )

    return wf
