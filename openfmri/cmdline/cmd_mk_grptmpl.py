# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the openfmri package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Create a group template image from functional fMRI data
"""

__docformat__ = 'restructuredtext'

# magic line for manpage summary
# man: -*- % cache all files required to generate the portal

import argparse
import os
import logging

from os.path import join as opj

from openfmri import cfg
from .helpers import parser_add_common_args, get_path_cfg, get_cfg_option, \
        arg2bool

lgr = logging.getLogger(__name__)
parser_args = dict(formatter_class=argparse.RawDescriptionHelpFormatter)


def setup_parser(parser):
    parser_add_common_args(parser,
        opt=('datadir', 'dataset', 'subjects', 'workdir'))
    parser_add_common_args(parser, required=True,
        opt=('task',))
    parser_add_common_args(parser, required=True, default='bold',
        opt=('flavor',))
    parser.add_argument(
        '--zpad', type=int, default=0,
        help="""Number of slices to add above and below the z-slice stack
        in the initial template image. This can aid alignment of images
        with small FOV in z-direction. Default: 0""")
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
    parser.add_argument('--apply-mni-sform', action='store_true',
        help="""If set, the template image's sform matrix will contain
        the estimated transformation into MNI space, i.e. mm-cordinates
        will be actual MNI coordinates. CAUTION: the current implementation
        only works properly for radiological images! (hence OFF by default)""")

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
            name='pick_best_reference',
            interface=Function(
                function=pick_closest_to_avg,
                input_names=['in_files', 'in_aux'],
                output_names=['out_brain', 'out_head']))
    wf.connect(brains, 'out', best_reference, 'in_files')
    wf.connect(heads, 'out', best_reference, 'in_aux')

    zpad_head = pe.Node(
            name='zpad_head',
            interface=Function(
                function=zslice_pad,
                input_names=['in_file', 'nslices'],
                output_names=['padded_file']))
    zpad_head.inputs.nslices = 20
    zpad_brain = zpad_head.clone('zpad_brain')
    if zpad > 0:
        wf.connect(best_reference, 'out_head', zpad_head, 'in_file')
        wf.connect(best_reference, 'out_brain', zpad_brain, 'in_file')

    # upsample 1stlvl reference
    # slight resolution bump for second iteration
    upsample_brain = pe.Node(
            name='upsample_init_brain',
            interface=afni.Resample(
                args='-rmode Cu -dxyz %s' % ' '.join(['%.2f' % i for i in target_resolution])))
    upsample_brain.inputs.out_file='upsampled_brain.nii.gz'
    upsample_brain.inputs.outputtype='NIFTI_GZ'
    upsample_head = upsample_brain.clone('upsample_init_head')
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


def make_subj_preproc_branch(task, wf, subj, datadir, datasrc):
    subj_sink = pe.Node(
            interface=nio.DataSink(
                parameterization=False,
                base_directory=os.path.abspath(datadir),
                container=subj,
                regexp_substitutions=[
                    ('/[^/]*\.nii', '.nii'),
                ]),
            name="subj_sink_%s" % subj,
            overwrite=True)

    # mean vol for each run
    make_samplevols = pe.MapNode(
            name='make_samplevol_%s' % subj,
    #        interface=fsl.ImageMaths(op_string='-Tmean'),
            interface=fsl.ExtractROI(t_min=0, t_size=1),
            iterfield=['in_file'])
    wf.connect(datasrc, 'bold_runs_%s' % subj, make_samplevols, 'in_file')
    # merge all samplevols to 4D for alignment
    merge_samplevols = pe.Node(
            name='merge_samplevols_%s' % subj,
            interface=fsl.Merge(dimension='t'))
    #wf.connect(make_samplevols, 'out_file', merge_samplevols, 'in_files')
    wf.connect(make_samplevols, 'roi_file', merge_samplevols, 'in_files')
    # 4D alignment across runs
    align_samplevols = pe.Node(
            name='align_samplevols_%s' % subj,
            interface=fsl.MCFLIRT(
                ref_vol=0,
                stages=4))
    wf.connect(merge_samplevols, 'merged_file', align_samplevols, 'in_file')
    wf.connect(align_samplevols, 'out_file',
               subj_sink, 'qa.%s.aligned_head_samples.@out' % task)
    # merge all aligned volumes into a normalized subject template
    make_subj_tmpl = pe.Node(
            name='make_subj_tmpl_%s' % subj,
            interface=Function(
                function=nonzero_normed_avg,
                input_names=['in_file'],
                output_names=['out_file', 'avg_stats']))
    wf.connect(align_samplevols, 'out_file', make_subj_tmpl, 'in_file')
    wf.connect(make_subj_tmpl, 'out_file',
               subj_sink, 'qa.%s.head.@out' % task)
    wf.connect(make_subj_tmpl, 'avg_stats',
               subj_sink, 'qa.%s.head_avgstats.@out' % task)
    # extract brain from subject template
    subj_bet = pe.Node(
            name='subj_bet_%s' % subj,
            interface=fsl.BET(
                  padding=True,
                  frac=0.15))
    wf.connect(make_subj_tmpl, 'out_file', subj_bet, 'in_file')
    wf.connect(subj_bet, 'out_file',
               subj_sink, 'qa.%s.brain.@out' % task)

    return subj_bet, make_subj_tmpl

def align_subj_to_tmpl_lin(wf, subj, lvl, brain_tmpl, head_tmpl, brain, head,
                       last_linear_align):
    align_brain_to_template = pe.Node(
            name='align_brain_to_template_lvl%i_%s' % (lvl, subj),
            interface=fsl.FLIRT(
                #no_search=True,
                #cost='normmi',
                #cost_func='normmi',
                uses_qform=True,
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
        name='project_head_to_template_lvl%i_%s' % (lvl, subj),
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

def align_subj_to_tmpl_nlin(wf, subj, lvl, brain_tmpl, head_tmpl, head, last_linear_align):
    # determine soft intersection mask
    soft_interssection = pe.Node(
            name='soft_intersection_lvl%i_%s' % (lvl, subj),
            interface=Function(
                function=max_thresh_mask,
                input_names=['in_file', 'mask_file', 'max_offset'],
                output_names=['out_file']))
    soft_interssection.inputs.max_offset = 0.8
    wf.connect(head_tmpl, 'avg_stats',
               soft_interssection, 'in_file')
    try:
        wf.connect(brain_tmpl, 'mask_file',
               soft_interssection, 'mask_file')
    except:
        wf.connect(brain_tmpl, 'avg_stats',
               soft_interssection, 'mask_file')

    # TODO: extend with inmask

    align_head_to_tmpl = pe.Node(
            name='align_head_to_tmpl_lvl%i_%s' % (lvl, subj),
            interface=fsl.FNIRT(
                intensity_mapping_model='global_non_linear_with_bias',
                #intensity_mapping_model='local_non_linear',
                warp_resolution=(10, 10, 10)))
                #warp_resolution=(7, 7, 7)))
    wf.connect(head, 'out_file',
               align_head_to_tmpl, 'in_file')
    wf.connect(soft_interssection, 'out_file',
               align_head_to_tmpl, 'refmask_file')
    wf.connect(head_tmpl, 'out_file',
               align_head_to_tmpl, 'ref_file')
    wf.connect(last_linear_align, 'out_matrix_file',
               align_head_to_tmpl, 'affine_file')

    return align_head_to_tmpl

def make_avg_template(wf, datasink, lvl, in_brains, in_heads,
                      linear):
    # merge and average for new template
    merge_heads = pe.Node(
                name='merge_heads_lvl%i' % lvl,
                interface=fsl.Merge(dimension='t'))
    wf.connect(merge_heads, 'merged_file',
               datasink, 'qa.lvl%i.aligned_head_samples.@out' % lvl)

    make_head_tmpl = pe.Node(
            name='make_head_tmpl_lvl%i' % lvl,
            interface=Function(
                function=nonzero_normed_avg,
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
                    name='merge_brains_lvl%i' % lvl,
                    interface=fsl.Merge(dimension='t'))
        wf.connect(merge_brains, 'merged_file',
                   datasink, 'qa.lvl%i.aligned_brain_samples.@out' % lvl)
        make_brain_tmpl = pe.Node(
                name='make_brain_tmpl_lvl%i' % lvl,
                interface=Function(
                    function=nonzero_normed_avg,
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
        tmpl_bet = pe.Node(
            name='tmpl_bet_lvl%i' % lvl,
            interface=fsl.BET(
                  padding=True,
                  mask=True,
                  frac=0.5))
        wf.connect(make_head_tmpl, 'out_file', tmpl_bet, 'in_file')
        wf.connect(tmpl_bet, 'out_file',
                   datasink, 'qa.lvl%i.brain.@out' % lvl)
        return tmpl_bet, make_head_tmpl

def trim_tmpl(wf, datasink, tmpl, name, roi):
    # trim the template
    x_min, x_size, y_min, y_size, z_min, z_size = roi
    trim_template = pe.Node(
            name='trim_%s' % name,
            interface=fsl.ExtractROI(
                #x_min=26, x_size=132,
                #y_min=7, y_size=175,
                #z_min=24,  z_size=48))
                x_min=x_min, x_size=x_size,
                y_min=y_min, y_size=y_size,
                z_min=z_min, z_size=z_size))
    wf.connect(tmpl, 'out_file', trim_template, 'in_file')
    ## final brain extract
    #bet = pe.Node(
    #        name='final_bet',
    #        interface=fsl.BET(
    #            padding=True,
    #            frac=0.55))
    #wf.connect(trim_template, 'roi_file', bet, 'in_file')
    return trim_template


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
            name='align2mni_7dof',
            interface=fsl.FLIRT(
                bins=256, cost='corratio',
                searchr_x=[-90, 90], searchr_y=[-90, 90],
                searchr_z=[-90, 90],
                dof=9,
                #searchr_x=[-20, 20], searchr_y=[-20, 20],
                #searchr_z=[-20, 20], dof=7,
                args="-interp trilinear",
                reference=mni_tmpl
            ))
    align2mni_12dof = pe.Node(
            name='align2mni_12dof',
            interface=fsl.FLIRT(
                cost='corratio',
                no_search=True,
                #searchr_x=[-90, 90], searchr_y=[-90, 90],
                #searchr_z=[-90, 90],
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
        wf.connect(mni_sform_brain, 'out_file', datasink, 'brain.@out')

    invert_xfm_epi2mni_12dof = pe.Node(
            name='invert_xfm_epi2mni_12dof',
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
            name='xfm_mni_12dof_in_epi_lin',
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
        wf.connect(mni_sform_brain, 'out_file',
                   xfm_mni_12dof_in_epi_lin, 'reference')
    else:
        wf.connect(brain_tmpl, 'out_file',
                   xfm_mni_12dof_in_epi_lin, 'reference')
    xfm_mni_12dof_in_epi_lin_masks = pe.MapNode(
            name='xfm_mni_12dof_in_epi_lin_masks',
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
        wf.connect(brain_tmpl, 'out_file',
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


def get_epi_tmpl_workflow(subjects,
                          dataset,
                          task,
                          target_resolution,
                          flavor,
                          datadir=os.path.abspath(os.curdir),
                          basedir=os.path.abspath(os.path.join(os.curdir, 'pipe_tmp')),
                          lin=3, nlin=2,
                          zpad=0,
                          template_roi=None,
                          set_mni_sform=False):
    # WORKFLOW
    wf = pe.Workflow(name="grptmpl_%s_%.3i_%s" % (dataset, task, flavor))
    wf.base_dir = os.path.abspath(basedir)
    wf.config['execution'] = {
        'stop_on_first_crash': 'True',}
    ##        'hash_method': 'timestamp'}

    # data source config
    data = nio.DataGrabber(
        outfields=['bold_runs_%s' % s for s in subjects]
    )
    data.inputs.template = '*'
    data.inputs.sort_filelist = False
    data.inputs.base_directory = os.path.abspath(datadir)
    data.inputs.field_template = dict(
        [('bold_runs_%s' %s, '%s/BOLD/task%.3i_run*/%s.nii*' % (s, task, flavor))
            for s in subjects]
        )
    datasrc = pe.Node(name='datasrc', interface=data)

    # data sink
    datasink = pe.Node(
        interface=nio.DataSink(
            parameterization=False,
            base_directory=os.path.abspath(os.path.join(datadir, 'templates')),
            container='task%.3i' % task,
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
                name='subjs_brains_lvl%i' % lvl,
                interface=util.Merge(len(subjects)))
        heads = pe.Node(
                name='subjs_heads_lvl%i' % lvl,
                interface=util.Merge(len(subjects)))
        if lvl == 0:
            brain_tmpl, head_tmpl = make_init_template(wf, datasink, lvl,
                                                       brains, heads,
                                                       target_resolution, zpad)
        else:
            brain_tmpl, head_tmpl = make_avg_template(
                                        wf, datasink, lvl, brains, heads,
                                        lvl <= lin)
        for i, subj in enumerate(subjects):
            if lvl == 0:
                brain, head = make_subj_preproc_branch('task%.3i' % task,
                                    wf, subj, datadir, datasrc)
                subj_nodes[subj] = (brain, head)
            else:
                # pull original nodes out of cache
                brain, head = subj_nodes[subj]
                # do linear alignment
                if lvl <= lin:
                    print 'linear'
                    brain, head = align_subj_to_tmpl_lin(
                        wf, subj, lvl,
                        latest_brain_tmpl, latest_head_tmpl,
                        brain, head, last_linear_align[subj])
                    last_linear_align[subj] = brain
                else:
                    print 'non-linear'
                    # non-linear alignment
                    head = align_subj_to_tmpl_nlin(
                            wf, subj, lvl,
                            latest_brain_tmpl,
                            latest_head_tmpl, head,
                            last_linear_align[subj])
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
        latest_brain_tmpl = trim_tmpl(wf, datasink, latest_brain_tmpl, 'brain', roi=template_roi)
        latest_head_tmpl = trim_tmpl(wf, datasink, latest_head_tmpl, 'head', roi=template_roi)
        out_slot = 'roi_file'
    else:
        out_slot = 'out_file'
    wf.connect(latest_brain_tmpl, out_slot, datasink, 'brain.@out')
    wf.connect(latest_head_tmpl, out_slot, datasink, 'head.@out')

    make_MNI_alignment(wf, mni_sink, latest_brain_tmpl, latest_head_tmpl,
                       set_mni_sform=set_mni_sform)
    return wf


def run(args):
    target_res = get_cfg_option('functional group template', 'resolution',
                               cli_input=args.target_resolution)
    if isinstance(target_res, basestring):
        target_res = [float(i) for i in target_res.split()]
    roi = get_cfg_option('functional group template', 'trim roi',
                        cli_input=args.trim_roi)
    if isinstance(roi, basestring):
        roi = [int(i) for i in roi.split()]

    datadir=get_path_cfg('common',
                        'data directory',
                        cli_input=args.datadir,
                        ensure_exists=True)
    datadir = opj(datadir, args.dataset)

    wf = get_epi_tmpl_workflow(['sub%s' % s for s in args.subjects],
                               args.dataset,
                               args.task,
                               target_res,
                               get_cfg_option('functional group template',
                                              'source data flavor',
                                              cli_input=args.flavor),
                               datadir,
                               basedir=get_path_cfg('common',
                                                    'working directory',
                                                    cli_input=args.workdir),
                               lin=2,
                               nlin=0,
                               zpad=get_cfg_option('functional group template',
                                                   'z-pad slices',
                                                   cli_input=args.zpad),
                               template_roi=roi,
                               set_mni_sform=arg2bool(
                                   get_cfg_option('functional group template',
                                                  'set mni sform',
                                                  cli_input=args.apply_mni_sform)))

#wf.write_graph(graph2use='flat')
    #wf.run(plugin='CondorDAGMan')
    wf.run()









