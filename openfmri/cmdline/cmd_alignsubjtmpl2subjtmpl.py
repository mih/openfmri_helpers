# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the openfmri package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Align two subject template image to each other
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
        opt=('datadir', 'dataset', 'subjects', 'workdir'))
    hlp.parser_add_common_args(parser, opt=('label',))
    parser.add_argument('--ref-template',
        help="""Reference template label""")
    parser.add_argument('--in-template',
        help="""Input template label""")

import sys
import os                                    # system functions
import nipype.interfaces.io as nio           # Data i/o		
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl

def align_template(wf, label, subj, intmpl, reftmpl, tmpldir):
    in_img = opj(tmpldir, intmpl, 'head.nii.gz')
    in_mask = opj(tmpldir, intmpl, 'brain_mask.nii.gz')
    ref_img = opj(tmpldir, reftmpl, 'head.nii.gz')
    ref_mask = opj(tmpldir, reftmpl, 'brain_mask.nii.gz')
    sink = pe.Node(
            interface=nio.DataSink(
                parameterization=False,
                base_directory=tmpldir,
                regexp_substitutions=[
                    ('/[^/]*\.nii', '.nii'),
                    ('/[^/]*\.mat', '.mat'),
                ]),
            name="sub%.3i_sink" % (subj,),
            overwrite=True)

    # alignment to template
    align = pe.Node(
            name='sub%.3i_align' % (subj,),
            interface=fsl.FLIRT(
                #cost='corratio',
                cost='mutualinfo',
                reference=ref_img,
                dof=6,
                interp='sinc'))
    align.inputs.in_file = in_img
    wf.connect(align, 'out_file',
               sink, '%s.in_%s.head.@out' % (intmpl, reftmpl))
    wf.connect(align, 'out_matrix_file',
               sink, '%s.in_%s.xfm_6dof.@out' % (intmpl, reftmpl))
    # project mask
    mask2ref = pe.Node(name='sub%.3i_mask2ref' % (subj,),
        interface=fsl.ApplyXfm(
            interp='nearestneighbour',
            reference=ref_img,
            apply_xfm=True))
    mask2ref.inputs.in_file = in_mask
    wf.connect(align, 'out_matrix_file', mask2ref, 'in_matrix_file')
    wf.connect(mask2ref, 'out_file',
               sink, '%s.in_%s.brain_mask.@out' % (intmpl, reftmpl))
    # invert XFM
    invert_xfm = pe.Node(
            name='sub%.3i_invert_xfm' % (subj,),
            interface=fsl.ConvertXFM(invert_xfm=True))
    wf.connect(align, 'out_matrix_file', invert_xfm, 'in_file')
    wf.connect(invert_xfm, 'out_file',
               sink, '%s.in_%s.xfm_6dof.@out' % (reftmpl, intmpl))
    # ref2tmpl
    ref2tmpl = pe.Node(name='sub%.3i_ref2tmpl' % (subj,),
        interface=fsl.ApplyXfm(
            interp='sinc',
            reference=in_img,
            apply_xfm=True))
    ref2tmpl.inputs.in_file = ref_img
    wf.connect(invert_xfm, 'out_file', ref2tmpl, 'in_matrix_file')
    wf.connect(ref2tmpl, 'out_file',
               sink, '%s.in_%s.head.@out' % (reftmpl, intmpl))
    # rev project mask
    refmask2tmpl = pe.Node(name='sub%.3i_refmask2tmpl' % (subj,),
        interface=fsl.ApplyXfm(
            interp='nearestneighbour',
            reference=in_img,
            apply_xfm=True))
    refmask2tmpl.inputs.in_file = ref_mask
    wf.connect(invert_xfm, 'out_file', refmask2tmpl, 'in_matrix_file')
    wf.connect(refmask2tmpl, 'out_file',
               sink, '%s.in_%s.brain_mask.@out' % (reftmpl, intmpl))
    return wf



def run(args):
    # label of the group template -- used to look up config options
    label = args.label
    cfg_section = 'alignsubjtmpl2subjtmpl %s' % label

    dataset = hlp.get_cfg_option('common', 'dataset', cli_input=args.dataset)

    subjects = hlp.get_dataset_subj_ids(args)
    subjects = hlp.exclude_subjects(subjects, cfg_section)

    dsdir = hlp.get_dataset_dir(args)

    wf_name = "alignsubjtmpl2subjtmpl_%s_%s" % (label, dataset)
    wf = hlp.get_base_workflow(wf_name.replace('.', '_'), args)

    ref_tmpl = hlp.get_cfg_option(cfg_section, 'reference template',
                                  cli_input=args.ref_template)
    in_tmpl = hlp.get_cfg_option(cfg_section,  'input template',
                                 cli_input=args.in_template)

    for subj in subjects:
        subj_tmpldir = os.path.abspath(opj(dsdir, 'sub%.3i' % subj,
                                           'templates'))
        wf = align_template(wf, label, subj, in_tmpl,
                            ref_tmpl, subj_tmpldir)
    return wf

