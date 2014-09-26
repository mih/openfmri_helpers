# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the openfmri package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Create a group template image
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
        opt=('datadir', 'dataset', 'subjects', 'workdir', 'input_expression', 'zslice_padding'))
    hlp.parser_add_common_args(parser, required=True,
        opt=('label',))
    parser.add_argument(
        '--linear', type=int,
        help="""Number of iterations of linear alignments""")
    parser.add_argument(
        '--nonlin', type=int,
        help="""Number of iterations of non-linear alignments. Experimental and
        disabled by default""")
    parser.add_argument(
        '--target-resolution', type=float, nargs=3, metavar='DIM',
        help="""Spatial target resolution of the group template in XYZ.
        This should be given in the same unit as the pixel dimensions
        of the imnput images. Default: keep resolution.""")
    parser.add_argument(
        '--trim-roi', type=int, nargs=6, metavar='INT',
        help="""ROI to trim the group space template to, given as
        x_min, x_size, y_min, y_size, z_min, z_size (voxel counts).
        Default: no trim""")
    parser.add_argument('--apply-mni-sform', type=hlp.arg2bool,
        help="""If set, the template image's sform matrix will contain
        the estimated transformation into MNI space, i.e. mm-cordinates
        will be actual MNI coordinates. CAUTION: the current implementation
        only works properly for radiological images! (hence OFF by default)""")
    parser.add_argument('--final-skullstrip', type=hlp.arg2bool,
        help="""If true, the final brain-extracted template image is created
        by a final skull-stripping of the head template image.""")
    parser.add_argument('--bet-padding', type=hlp.arg2bool,
        help="""Enable padding for BET""")
    parser.add_argument('--tmpl-bet-frac', type=float,
        help="""Frac parameter for the template skullstripping""")
    parser.add_argument('--tmpl-bet-gradient', type=float,
        help="""Gradient parameter for the template skullstripping""")
    parser.add_argument('--subj-bet-frac', type=float,
        help="""Frac parameter for skullstripping subject images""")
    parser.add_argument('--nonlin-mapping-model',
        choices=('none', 'global_linear', 'global_non_linear', 'local_linear',
                 'global_non_linear_with_bias', 'local_non_linear'),
        help="""Intensity mapping model for FNIRT""")
    parser.add_argument('--use-qform', type=hlp.arg2bool,
        help="""Whether to pass the -usesqform flag to FLIRT""")
    parser.add_argument('--initial-reference-brain', metavar='FILE',
        help="""Skull-stripped brain volume to use as an initial alignment.
        If not specified, one of the input images is used as a reference
        (the one with the smallest absolute difference from the mean of all
        input images).""")

import sys
import os                                    # system functions
import nipype.interfaces.io as nio           # Data i/o		
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
import nipype.interfaces.afni as afni
from nipype.interfaces.utility import Function
from ..nipype_helpers import *

def make_init_template(wf, datasink, lvl, brains, heads, target_resolution,
                       zpad=0):
    # pick one subject as the 1st-level reference
    best_reference = pe.Node(
            name='lvl%i_pick_best_reference' % lvl,
            interface=Function(
                function=pick_closest_to_avg,
                input_names=['in_files', 'in_aux'],
                output_names=['out_brain', 'out_head']))
    wf.connect(brains, 'out', best_reference, 'in_files')
    wf.connect(heads, 'out', best_reference, 'in_aux')

    zpad_head = pe.Node(
            name='lvl%i_zpad_head' % lvl,
            interface=Function(
                function=zslice_pad,
                input_names=['in_file', 'nslices'],
                output_names=['padded_file']))
    zpad_head.inputs.nslices = zpad
    zpad_brain = zpad_head.clone('lvl%i_zpad_brain' % lvl)
    if zpad > 0:
        wf.connect(best_reference, 'out_head', zpad_head, 'in_file')
        wf.connect(best_reference, 'out_brain', zpad_brain, 'in_file')

    # upsample 1stlvl reference
    # slight resolution bump for second iteration
    upsample_brain = pe.Node(
            name='lvl%i_upsample_init_brain' % lvl,
            interface=afni.Resample(
                args='-rmode Cu -dxyz %s' % ' '.join(['%.2f' % i for i in target_resolution])))
    upsample_brain.inputs.out_file='upsampled_brain.nii.gz'
    upsample_brain.inputs.outputtype='NIFTI_GZ'
    upsample_head = upsample_brain.clone('lvl%i_upsample_init_head' % lvl)
    upsample_head.inputs.out_file='upsampled_head.nii.gz'
    if zpad > 0:
        wf.connect(zpad_brain, 'padded_file', upsample_brain, 'in_file')
        wf.connect(zpad_head, 'padded_file', upsample_head, 'in_file')
    else:
        wf.connect(best_reference, 'out_head', upsample_head, 'in_file')
        wf.connect(best_reference, 'out_brain', upsample_brain, 'in_file')

    wf.connect(upsample_brain, 'out_file', datasink, 'qa.lvl%i.brain.@out' % lvl)
    wf.connect(upsample_head, 'out_file', datasink, 'qa.lvl%i.head.@out' % lvl)

    return upsample_brain, upsample_head


def make_subj_preproc_branch(label, wf, subj, datadir, datasrc, 
        bet_frac=0.5, bet_padding=False):
    subj_sink = pe.Node(
            interface=nio.DataSink(
                parameterization=False,
                base_directory=os.path.abspath(datadir),
                container='sub%.3i' % subj,
                regexp_substitutions=[
                    ('/[^/]*\.nii', '.nii'),
                ]),
            name="sub%.3i_sink" % subj,
            overwrite=True)

    # mean vol for each run
    make_samplevols = pe.MapNode(
            name='sub%.3i_make_samplevol' % subj,
    #        interface=fsl.ImageMaths(op_string='-Tmean'),
            interface=fsl.ExtractROI(t_min=0, t_size=1),
            iterfield=['in_file'])
    wf.connect(datasrc, 'out_paths', make_samplevols, 'in_file')
    # merge all samplevols to 4D for alignment
    merge_samplevols = pe.Node(
            name='sub%.3i_merge_samplevols' % subj,
            interface=fsl.Merge(dimension='t'))
    #wf.connect(make_samplevols, 'out_file', merge_samplevols, 'in_files')
    wf.connect(make_samplevols, 'roi_file', merge_samplevols, 'in_files')
    # normalize all input images
    normalize_vols = pe.Node(
            name='sub%.3i_normalize_samplevols' % subj,
            interface=fsl.ImageMaths(op_string='-inm 1000'))
    wf.connect(merge_samplevols, 'merged_file', normalize_vols, 'in_file')
    # 4D alignment across runs
    align_samplevols = pe.Node(
            name='sub%.3i_align_samplevols' % subj,
            interface=fsl.MCFLIRT(
                ref_vol=0,
                stages=4))
    wf.connect(normalize_vols, 'out_file', align_samplevols, 'in_file')
    wf.connect(align_samplevols, 'out_file',
               subj_sink, 'qa.%s.aligned_head_samples.@out' % label)
    # merge all aligned volumes into a normalized subject template
    make_subj_tmpl = pe.Node(
            name='sub%.3i_make_subj_tmpl' % subj,
            interface=Function(
                function=nonzero_avg,
                input_names=['in_file'],
                output_names=['out_file', 'avg_stats']))
    wf.connect(align_samplevols, 'out_file', make_subj_tmpl, 'in_file')
    wf.connect(make_subj_tmpl, 'out_file',
               subj_sink, 'qa.%s.head.@out' % label)
    wf.connect(make_subj_tmpl, 'avg_stats',
               subj_sink, 'qa.%s.head_avgstats.@out' % label)
    # extract brain from subject template
    if bet_padding:
        bet_interface=fsl.BET(padding=True, frac=bet_frac)
    else:
        bet_interface=fsl.BET(robust=True, frac=bet_frac)
    subj_bet = pe.Node(
            name='sub%.3i_bet' % subj,
            interface=bet_interface)
    wf.connect(make_subj_tmpl, 'out_file', subj_bet, 'in_file')
    wf.connect(subj_bet, 'out_file',
               subj_sink, 'qa.%s.brain.@out' % label)

    return subj_bet, make_subj_tmpl

def align_subj_to_tmpl_lin(wf, subj, lvl, brain_tmpl, head_tmpl, brain, head,
                       last_linear_align, use_qform=True):
    align_brain_to_template = pe.Node(
            name='sub%.3i_align_brain_to_template_lvl%i' % (subj, lvl),
            interface=fsl.FLIRT(
                #no_search=True,
                #cost='normmi',
                #cost_func='normmi',
                uses_qform=use_qform,
                searchr_x=[-90, 90],
                searchr_y=[-90, 90],
                searchr_z=[-90, 90],
                dof=12,
                args="-interp trilinear"))
    if not last_linear_align is None:
        wf.connect(last_linear_align, 'out_matrix_file',
                   align_brain_to_template, 'in_matrix_file')
        #wf.connect(brain_tmpl, 'mask_file',
        #           align_brain_to_template, 'ref_weight')
    wf.connect(brain, 'out_file', align_brain_to_template, 'in_file')
    wf.connect(brain_tmpl, 'out_file', align_brain_to_template, 'reference')

    # project the subject's head image onto the 1st-level template
    project_head_to_template = pe.Node(
        name='sub%.3i_project_head_to_template_lvl%i' % (subj, lvl),
        interface=fsl.ApplyXfm(
            interp='trilinear',
            apply_xfm=True))
    wf.connect(align_brain_to_template, 'out_matrix_file',
               project_head_to_template, 'in_matrix_file')
    wf.connect(head, 'out_file',
               project_head_to_template, 'in_file')
    wf.connect(head_tmpl, 'out_file',
               project_head_to_template, 'reference')

    return align_brain_to_template, project_head_to_template

def align_subj_to_tmpl_nlin(wf, subj, lvl, brain_tmpl, head_tmpl, head,
                            last_linear_align, intmod):
    align_head_to_tmpl = pe.Node(
            name='sub%.3i_align_head_to_tmpl_lvl%i' % (subj, lvl),
            interface=fsl.FNIRT(
                intensity_mapping_model=intmod,
                #intensity_mapping_model='global_non_linear_with_bias',
                #intensity_mapping_model='local_non_linear',
                warp_resolution=(10, 10, 10)))
                #warp_resolution=(7, 7, 7)))
    wf.connect(head, 'out_file',
               align_head_to_tmpl, 'in_file')
    wf.connect(head_tmpl, 'out_file',
               align_head_to_tmpl, 'ref_file')
    wf.connect(last_linear_align, 'out_matrix_file',
               align_head_to_tmpl, 'affine_file')

    return align_head_to_tmpl

def make_avg_template(wf, datasink, lvl, in_brains, in_heads,
                      linear, bet_frac=0.5, bet_padding=False):
    # merge and average for new template
    merge_heads = pe.Node(
                name='lvl%i_merge_heads' % lvl,
                interface=fsl.Merge(dimension='t'))
    wf.connect(merge_heads, 'merged_file',
               datasink, 'qa.lvl%i.aligned_head_samples.@out' % lvl)

    make_head_tmpl = pe.Node(
            name='lvl%i_make_head_tmpl' % lvl,
            interface=Function(
                function=nonzero_avg,
                input_names=['in_file'],
                output_names=['out_file', 'avg_stats']))
    wf.connect(in_heads, 'out', merge_heads, 'in_files')
    wf.connect(merge_heads, 'merged_file', make_head_tmpl, 'in_file')
    wf.connect(make_head_tmpl, 'out_file',
               datasink, 'qa.lvl%i.head.@out' % lvl)
    wf.connect(make_head_tmpl, 'avg_stats',
               datasink, 'qa.lvl%i.head_avgstats.@out' % lvl)

    if linear:
        merge_brains = pe.Node(
                    name='lvl%i_merge_brains' % lvl,
                    interface=fsl.Merge(dimension='t'))
        wf.connect(merge_brains, 'merged_file',
                   datasink, 'qa.lvl%i.aligned_brain_samples.@out' % lvl)
        make_brain_tmpl = pe.Node(
                name='lvl%i_make_brain_tmpl' % lvl,
                interface=Function(
                    function=nonzero_avg,
                    input_names=['in_file'],
                    output_names=['out_file', 'avg_stats']))
        wf.connect(in_brains, 'out', merge_brains, 'in_files')
        wf.connect(merge_brains, 'merged_file', make_brain_tmpl, 'in_file')
        wf.connect(make_brain_tmpl, 'out_file',
                   datasink, 'qa.lvl%i.brain.@out' % lvl)
        wf.connect(make_brain_tmpl, 'avg_stats',
                   datasink, 'qa.lvl%i.brain_avgstats.@out' % lvl)
        return make_brain_tmpl, make_head_tmpl
    else:
        # skull-strip the head template
        if bet_padding:
            bet_interface=fsl.BET(padding=True, frac=bet_frac)
        else:
            bet_interface=fsl.BET(robust=True, frac=bet_frac)
        tmpl_bet = pe.Node(
            name='lvl%i_tmpl_bet' % lvl,
            interface=bet_interface)
        wf.connect(make_head_tmpl, 'out_file', tmpl_bet, 'in_file')
        wf.connect(tmpl_bet, 'out_file',
                   datasink, 'qa.lvl%i.brain.@out' % lvl)
        return tmpl_bet, make_head_tmpl

def trim_tmpl(wf, tmpl, name, roi):
    # trim the template
    x_min, x_size, y_min, y_size, z_min, z_size = roi
    trim_template = pe.Node(
            name='trim_%s' % name,
            interface=fsl.ExtractROI(
                x_min=x_min, x_size=x_size,
                y_min=y_min, y_size=y_size,
                z_min=z_min, z_size=z_size))
    wf.connect(tmpl, 'out_file', trim_template, 'in_file')
    return trim_template

def final_bet(wf, head_tmpl, slot_name, padding=False, frac=0.5,
              gradient=0):
    if padding:
        interface=fsl.BET(padding=True, frac=frac, vertical_gradient=gradient)
    else:
        interface=fsl.BET(robust=True, frac=frac, vertical_gradient=gradient)
    bet = pe.Node(name='final_bet', interface=interface)
    wf.connect(head_tmpl, slot_name, bet, 'in_file')
    return bet


def make_MNI_alignment(wf, datasink, brain_tmpl, head_tmpl, set_mni_sform=True):
    if hasattr(brain_tmpl.outputs, 'roi_file'):
        out_slot = 'roi_file'
    else:
        out_slot = 'out_file'
    # now align with MNI152
    mni_tmpl='/usr/share/data/fsl-mni152-templates/MNI152_T1_1mm_brain.nii.gz'
    # only rigid body plus rescale -- the average should be a pretty good
    # match with MNI -- deals with small brain coverage
    align2mni_9dof = pe.Node(
            name='mni_tmplalign_9dof',
            interface=fsl.FLIRT(
                bins=256, cost='corratio',
                searchr_x=[-90, 90], searchr_y=[-90, 90],
                searchr_z=[-90, 90],
                dof=9,
                args="-interp trilinear",
                reference=mni_tmpl
            ))
    align2mni_12dof = pe.Node(
            name='mni_tmplalign_12dof',
            interface=fsl.FLIRT(
                cost='corratio',
                no_search=True,
                dof=12,
                args="-interp trilinear",
                reference=mni_tmpl,
            ))
    wf.connect(brain_tmpl, out_slot, align2mni_9dof, 'in_file')
    wf.connect(brain_tmpl, out_slot, align2mni_12dof, 'in_file')
    # use the 7DOF output as reference weight for the 12DOF alignment
    wf.connect(align2mni_9dof, 'out_file', align2mni_12dof, 'ref_weight')
    wf.connect(align2mni_9dof, 'out_matrix_file', align2mni_12dof, 'in_matrix_file')

    mni_sform_brain = pe.Node(
            name='mni_sform_brain',
            interface=Function(
                function=update_sform2mni,
                input_names=['in_file', 'xfm', 'reference'],
                output_names=['out_file']))
    mni_sform_brain.inputs.reference = mni_tmpl
    if set_mni_sform:
        wf.connect(brain_tmpl, out_slot, mni_sform_brain, 'in_file')
        wf.connect(align2mni_12dof, 'out_matrix_file', mni_sform_brain, 'xfm')
        wf.connect(mni_sform_brain, 'out_file', datasink, 'brain.@mni')

    invert_xfm_epi2mni_12dof = pe.Node(
            name='mni_invert_tmplalign_xfm',
            interface=fsl.ConvertXFM(invert_xfm=True))
    wf.connect(align2mni_12dof, 'out_matrix_file',
               invert_xfm_epi2mni_12dof, 'in_file')
    wf.connect([
        (align2mni_12dof, datasink, [
            ('out_file', 'in_mni.brain_12dof.@out'),
            ('out_matrix_file', 'xfm.tmpl2mni_12dof.@out'),
        ]),
        (invert_xfm_epi2mni_12dof, datasink, [
            ('out_file', 'xfm.mni2tmpl_12dof.@out'),
        ]),
    ])
    # grab MNI152 property images and project them back
    xfm_mni_12dof_in_epi_lin = pe.MapNode(
            name='mni_xfm_affine',
            interface=fsl.ApplyXfm(
                args="-interp trilinear",
                apply_xfm=True),
            iterfield=['in_file'])
    xfm_mni_12dof_in_epi_lin.inputs.in_file = [
            '/usr/share/data/fsl-mni152-templates/MNI152_T1_1mm.nii.gz',
            '/usr/share/data/fsl-mni152-templates/tissuepriors/avg152T1_brain.hdr',
            '/usr/share/data/fsl-mni152-templates/tissuepriors/avg152T1_csf.hdr',
            '/usr/share/data/fsl-mni152-templates/tissuepriors/avg152T1_gray.hdr',
            '/usr/share/data/fsl-mni152-templates/tissuepriors/avg152T1_white.hdr',
        ]
    wf.connect(invert_xfm_epi2mni_12dof, 'out_file',
               xfm_mni_12dof_in_epi_lin, 'in_matrix_file')
    if set_mni_sform:
        wf.connect(mni_sform_brain, out_slot,
                   xfm_mni_12dof_in_epi_lin, 'reference')
    else:
        wf.connect(brain_tmpl, out_slot,
                   xfm_mni_12dof_in_epi_lin, 'reference')
    xfm_mni_12dof_in_epi_lin_masks = pe.MapNode(
            name='mni_xfm_affine_mask',
            interface=fsl.ApplyXfm(
                args="-interp trilinear",
                apply_xfm=True),
            iterfield=['in_file'])
    xfm_mni_12dof_in_epi_lin_masks.inputs.in_file = [
            '/usr/share/data/fsl-mni152-templates/MNI152_T1_1mm.nii.gz',
            '/usr/share/data/fsl-mni152-templates/MNI152_T1_1mm_brain_mask.nii.gz',
            '/usr/share/data/fsl-mni152-templates/MNI152_T1_1mm_brain_mask_dil.nii.gz',
        ]
    xfm_mni_12dof_in_epi_lin_masks.inputs.interp='nearestneighbour'
    wf.connect(invert_xfm_epi2mni_12dof, 'out_file',
               xfm_mni_12dof_in_epi_lin_masks, 'in_matrix_file')
    if set_mni_sform:
        wf.connect(mni_sform_brain, 'out_file',
                   xfm_mni_12dof_in_epi_lin_masks, 'reference')
    else:
        wf.connect(brain_tmpl, out_slot,
                   xfm_mni_12dof_in_epi_lin_masks, 'reference')

    # datasink connections
    wf.connect([
        (xfm_mni_12dof_in_epi_lin, datasink, [
            ('out_file', 'from_mni.@metric'),
        ]),
        (xfm_mni_12dof_in_epi_lin_masks, datasink, [
            ('out_file', 'from_mni.@masks'),
        ]),
    ])


def get_epi_tmpl_workflow(wf, datasrcs,
                          subjects,
                          label,
                          target_resolution,
                          datadir=os.path.abspath(os.curdir),
                          lin=3, nlin=1,
                          zpad=0,
                          template_roi=None,
                          set_mni_sform=False,
                          do_final_bet=False,
                          bet_padding=False,
                          subj_bet_frac=0.5,
                          tmpl_bet_frac=0.5,
                          tmpl_bet_gradient=0,
                          intmod='global_non_linear_with_bias',
                          use_qform=True,
                          init_template=None
                          ):
    # data sink
    datasink = pe.Node(
        interface=nio.DataSink(
            parameterization=False,
            base_directory=os.path.abspath(os.path.join(datadir, 'templates')),
            container=label,
            regexp_substitutions=[
                ('/[^/]*\.nii', '.nii'),
            ]),
        name="datasink",
        overwrite=True)

    # collect end nodes to merge across subjects
    subj_nodes = {}
    latest_brain_tmpl = None
    latest_head_tmpl = None
    last_linear_align = dict(zip(subjects, [None] * len(subjects)))
    for lvl in range(lin + nlin + 1):
        print 'Level %i' % lvl
        brains = pe.Node(
                name='lvl%i_subjs_brains' % lvl,
                interface=util.Merge(len(subjects)))
        heads = pe.Node(
                name='lvl%i_subjs_heads' % lvl,
                interface=util.Merge(len(subjects)))
        if lvl == 0:
            if init_template is None:
                b, h = (brains, heads)
            else:
                if not os.path.isabs(init_template):
                    init_template = opj(datadir, init_template)
                b = pe.Node(
                        name='lvl0_init_templ_brain',
                        interface=util.IdentityInterface(
                            fields=['out']))
                b.inputs.out = [init_template]
                h = b.clone('lvl0_init_templ_head')
            brain_tmpl, head_tmpl = make_init_template(wf, datasink, lvl,
                                                       b, h,
                                                       target_resolution, zpad)
        else:
            brain_tmpl, head_tmpl = make_avg_template(
                                        wf, datasink, lvl, brains, heads,
                                        lvl <= lin,
                                        bet_frac=tmpl_bet_frac,
                                        bet_padding=bet_padding)
        for i, subj in enumerate(subjects):
            if lvl == 0:
                brain, head = make_subj_preproc_branch('template_%s' % label,
                                    wf, subj, datadir, datasrcs[subj],
                                    bet_frac=subj_bet_frac,
                                    bet_padding=bet_padding)
                subj_nodes[subj] = (brain, head)
            else:
                # pull original nodes out of cache
                brain, head = subj_nodes[subj]
                # do linear alignment
                if lvl <= lin:
                    brain, head = align_subj_to_tmpl_lin(
                        wf, subj, lvl,
                        latest_brain_tmpl, latest_head_tmpl,
                        brain, head, last_linear_align[subj],
                        use_qform=use_qform)
                    last_linear_align[subj] = brain
                else:
                    # non-linear alignment
                    head = align_subj_to_tmpl_nlin(
                            wf, subj, lvl,
                            latest_brain_tmpl,
                            latest_head_tmpl, head,
                            last_linear_align[subj],
                            intmod)
            wf.connect(brain, 'out_file', brains, 'in%i' % (i + 1))
            try:
                wf.connect(head, 'warped_file', heads, 'in%i' % (i + 1))
            except:
                wf.connect(head, 'out_file', heads, 'in%i' % (i + 1))
        latest_brain_tmpl = brain_tmpl
        latest_head_tmpl = head_tmpl

    mni_sink = datasink.clone('mni_datasink')
    mni_sink.inputs.regexp_substitutions=[
            ('_flirt', ''),
            ('brain/updated_sform\.nii', 'brain.nii'),
            ('head/updated_sform\.nii', 'head.nii'),
            ('_12dof/[^\.]*\.', '_12dof.'),
            ]
    if not template_roi is None:
        latest_brain_tmpl = trim_tmpl(wf, latest_brain_tmpl, 'brain', roi=template_roi)
        latest_head_tmpl = trim_tmpl(wf, latest_head_tmpl, 'head', roi=template_roi)
        out_slot = 'roi_file'
    else:
        out_slot = 'out_file'
    if do_final_bet:
        latest_brain_tmpl = final_bet(wf, latest_head_tmpl, out_slot,
                                      bet_padding, tmpl_bet_frac,
                                      tmpl_bet_gradient)
        if not set_mni_sform:
            wf.connect(latest_brain_tmpl, 'out_file', datasink, 'brain.@out')
    else:
        if not set_mni_sform:
            wf.connect(latest_brain_tmpl, out_slot, datasink, 'brain.@out')
    wf.connect(latest_head_tmpl, out_slot, datasink, 'head.@out')

    make_MNI_alignment(wf, mni_sink, latest_brain_tmpl, latest_head_tmpl,
                       set_mni_sform=set_mni_sform)
    return wf


def run(args):
    # label of the group template -- used to look up config options
    label = args.label
    cfg_section = 'group template %s' % label

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

    wf_name = "grptmpl_%s_%s" % (label, dataset)
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

    wf = get_epi_tmpl_workflow(wf, datasrcs, subjects, label,
            target_res,
            dsdir,
            lin=int(hlp.get_cfg_option(cfg_section,
                                       'linear iterations',
                                       cli_input=args.linear,
                                       default=1)),
            nlin=int(hlp.get_cfg_option(cfg_section,
                                        'non-linear iterations',
                                        cli_input=args.nonlin,
                                        default=0)),
            zpad=int(hlp.get_cfg_option(cfg_section,
                                        'zslice padding',
                                        cli_input=args.zslice_padding,
                                        default=0)),
            template_roi=roi,
            set_mni_sform=hlp.arg2bool(
                            hlp.get_cfg_option(
                                cfg_section,
                                'apply mni sform',
                                cli_input=args.apply_mni_sform,
                                default=False)),
            do_final_bet=hlp.arg2bool(
                        hlp.get_cfg_option(
                            cfg_section,
                            'final skullstrip',
                            cli_input=args.final_skullstrip,
                            default=False)),
            bet_padding=hlp.arg2bool(
                        hlp.get_cfg_option(
                            cfg_section,
                            'bet padding',
                            cli_input=args.bet_padding,
                            default=False)),
            subj_bet_frac=float(
                        hlp.get_cfg_option(
                            cfg_section,
                            'subject bet frac',
                            cli_input=args.subj_bet_frac,
                            default=0.5)),
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
            intmod=hlp.get_cfg_option(
                            cfg_section,
                            'nonlin intensity mapping',
                            cli_input=args.nonlin_mapping_model,
                            default='global_non_linear_with_bias'),
            use_qform=hlp.arg2bool(
                        hlp.get_cfg_option(
                            cfg_section,
                            'use qform',
                            cli_input=args.use_qform,
                            default=True)),
            init_template=hlp.get_cfg_option(
                            cfg_section,
                            'initial reference brain',
                            cli_input=args.initial_reference_brain,
                            default=None),
            )

    return wf
