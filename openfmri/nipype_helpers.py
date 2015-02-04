def make_meanvol_brain(bold_run):
    import nipype.interfaces.fsl as fsl
    import os
    outname = os.path.basename(bold_run)[:-7]
    # reference volume is the mean vol
    meanvol = fsl.ImageMaths(in_file=bold_run, op_string= '-Tmean',
                             out_file='%s_meanvol.nii.gz' % outname).run()
    # skull strip individually
    bet = fsl.BET(in_file='%s_meanvol.nii.gz' % outname,
                  out_file='%s_meanvol_brain.nii.gz' % outname,
                  padding=True, mask=True, frac=0.05).run()
    return [os.path.abspath(i) for i in (
        '%s_meanvol.nii.gz' % outname,
        '%s_meanvol_brain.nii.gz' % outname,
        '%s_meanvol_brain_mask.nii.gz' % outname,
        )]

def pick_closest_to_avg(in_files, in_aux):
    import os
    import numpy as np
    import nibabel as nb
    import nipype.interfaces.fsl as fsl
    from openfmri.nipype_helpers import nonzero_avg
    # send timeseries to normed avg
    fsl.Merge(dimension='t',
              in_files=in_files,
              merged_file='merged_input.nii.gz').run()
    avg, avg_stats = nonzero_avg('merged_input.nii.gz')
    avg_data = nb.load(avg).get_data()
    dists = [np.sum((avg_data - nb.load(vol).get_data())**2) for vol in in_files]
    min_idx = np.argmin(dists)
    if len(in_aux) == len(in_files):
        return os.path.abspath(in_files[min_idx]), \
               os.path.abspath(in_aux[min_idx])
    else:
        return os.path.abspath(in_files[min_idx]), None


def zslice_pad(in_file, nslices):
    import os
    if not nslices:
        return os.path.abspath(in_file)
    from subprocess import check_call
    check_call(['3dZeropad', '-IS', str(nslices),
                '-prefix', 'padded.nii.gz', in_file])
    return os.path.abspath('padded.nii.gz')

def nonzero_avg(in_file):
    import numpy as np
    import os
    import nibabel as nb
    in_img = nb.load(in_file)
    in_data = in_img.get_data()
    avg_data = np.zeros(in_data.shape[:3], dtype=np.float)
    avg_count = np.zeros(in_data.shape[:3], dtype=np.int)
    if len(in_data.shape) > 3:
        for i, vol in enumerate(np.rollaxis(in_data, 3, 0)):
            avg_data += vol
            avg_count += vol > 0
    else:
        avg_data = in_data
    nonzero = avg_count > 0
    avg_data[nonzero] /= avg_count[nonzero]
    nb.save(nb.Nifti1Image(avg_data, in_img.get_affine()),
            'avg.nii.gz')
    nb.save(nb.Nifti1Image(avg_count, in_img.get_affine()),
            'avg_overlap.nii.gz')
    return [os.path.abspath(i) for i in (
        'avg.nii.gz',
        'avg_overlap.nii.gz',
        )]

def make_epi_template(bold_brains):
    import numpy as np
    import os
    import nibabel as nb
    import nipype.interfaces.fsl as fsl
    import nipype.interfaces.afni as afni
    tmpl1st_name = bold_brains[0]
    tmpl1st_img = nb.load(tmpl1st_name)
    tmpl1st_data = tmpl1st_img.get_data()
    tmpl1st_count = (tmpl1st_data > 1).astype(np.int16)
    # align all brains to the first one
    to_merge = []
    for i, brain in enumerate(bold_brains):
        fsl.ImageMaths(in_file=brain,
                       op_string='-inm 1000',
                       out_file='%i_brain_norm.nii.gz' % i).run()
        aligned_fname = '%i_brain_in_tmpl1st.nii.gz' % i
        fsl.FLIRT(
            bins=256, cost_func='normcorr',
            no_search=True,
#            searchr_x=[-5, 5], searchr_y=[-5, 5],
#            searchr_z=[-5, 5],
            dof=12, interp='sinc',
            reference=tmpl1st_name,
            in_file='%i_brain_norm.nii.gz' % i,
            out_file=aligned_fname).run()
        # add to average
        aligned1st = nb.load(aligned_fname).get_data()
        tmpl1st_data += aligned1st
        tmpl1st_count += aligned1st > 1
        # store for QA merge
        to_merge.append(aligned_fname)
    # 2nd level template
    tmpl1st_data /= tmpl1st_count
    nb.save(nb.Nifti1Image(tmpl1st_data, tmpl1st_img.get_affine()),
            'epi_tmpl2nd_coarse.nii.gz')
    # merge aligned files for QA
    fsl.Merge(dimension='t', in_files=to_merge,
              merged_file='1st_level_alignment_qa.nii.gz').run()
    # slight resolution bump for second iteration
    afni.Resample(in_file='epi_tmpl2nd_coarse.nii.gz',
                  out_file='epi_tmpl2nd.nii.gz',
                  args='-dxyz 1.2 1.2 1.2').run()
    tmpl_img = nb.load('epi_tmpl2nd.nii.gz')
    tmpl_data = np.zeros(tmpl_img.get_shape(), dtype=np.float64)
    tmpl_count = np.zeros(tmpl_img.get_shape(), dtype=np.int16)
    to_merge = []
    for i, brain in enumerate(bold_brains):
        aligned_fname = '%i_brain_in_tmpl2nd.nii.gz' % i
        interface=fsl.FLIRT(
            bins=256, cost_func='normcorr',
            no_search=True,
            #searchr_x=[-5, 5], searchr_y=[-5, 5],
            #searchr_z=[-5, 5],
            dof=12, interp='sinc',
            reference='epi_tmpl2nd.nii.gz',
            in_file='%i_brain_norm.nii.gz' % i,
            out_file=aligned_fname).run()
        # add to average
        aligned2nd = nb.load(aligned_fname).get_data()
        tmpl_data += aligned2nd
        tmpl_count += aligned2nd > 1
        # store for QA merge
        to_merge.append(aligned_fname)
    tmpl_data /= tmpl_count
    nb.save(nb.Nifti1Image(tmpl_data, tmpl_img.get_affine()),
            'epi_tmplfull.nii.gz')
    nb.save(nb.Nifti1Image(tmpl_count, tmpl_img.get_affine()),
            'epi_tmploverlap.nii.gz')
    # merge aligned files for QA
    fsl.Merge(dimension='t', in_files=to_merge,
              merged_file='2nd_level_alignment_qa.nii.gz').run()
    open('2nd_level_qa_file_order.txt', 'w').write(
            '\n'.join(to_merge))
    stripair = fsl.ExtractROI(
                    in_file='epi_tmplfull.nii.gz',
                    roi_file='epi_tmpltight.nii.gz',
                    x_min=30, x_size=125,
                    y_min=15, y_size=155,
                    z_min=0,  z_size=45,
                    ).run()
    return [os.path.abspath(i) for i in (
        'epi_tmplfull.nii.gz',
        'epi_tmpltight.nii.gz',
        'epi_tmpl2nd.nii.gz',
        'epi_tmploverlap.nii.gz',
        '1st_level_alignment_qa.nii.gz',
        '2nd_level_alignment_qa.nii.gz',
        '2nd_level_qa_file_order.txt',
        )]

def max_thresh_mask(in_file, mask_file, max_offset):
    import os
    import numpy as np
    import nibabel as nb
    in_img = nb.load(in_file)
    in_data = in_img.get_data()
    #mask_data = nb.load(mask_file).get_data().astype(int)
    out_data = (in_data > (float(in_data.max()) * max_offset)).astype(np.int16)
    #out_data *= mask_data
    nb.Nifti1Image(out_data, in_img.get_affine()).to_filename('maxthresh_mask.nii.gz')
    return os.path.abspath('maxthresh_mask.nii.gz')

def update_sform2mni(in_file, xfm, reference):
    import os
    import numpy as np
    import nibabel as nb
    in_img = nb.load(in_file)
    in_hdr = in_img.get_header()
    xfm_mat = np.matrix(np.loadtxt(xfm))
    ref_hdr = nb.load(reference).get_header()
    in_img = nb.load(in_file)
    sform = ref_hdr.get_best_affine() \
            * np.matrix(np.diag(ref_hdr.get_zooms() + (1,))).I \
            * xfm_mat \
            * np.diag(in_hdr.get_zooms() + (1,))
    in_hdr.set_sform(sform, code='mni')
    nb.Nifti1Image(in_img.get_data(), sform, in_hdr).to_filename('updated_sform.nii.gz')
    return os.path.abspath('updated_sform.nii.gz')

def fix_xfm_after_zpad(xfm, reference, nslices):
    import os
    import numpy as np
    import nibabel as nb
    ref_hdr = nb.load(reference).get_header()
    xfm_mat = np.matrix(np.loadtxt(xfm))
    zpixdim = ref_hdr.get_zooms()[-1]
    xfm_mat[2,3] -= nslices * zpixdim
    np.savetxt('fixed.mat', xfm_mat)
    return os.path.abspath('fixed.mat')

def tsnr_from_filename(in_file):
    import nibabel as nb
    import numpy as np
    import os

    outname = 'tsnr.nii.gz'

    img = nb.load(in_file)
    data = img.get_data()
    mean_d = np.mean(data,axis=-1)
    std_d = np.std(data,axis=-1)
    tsnr = np.ma.array(mean_d, mask=std_d == 0) / std_d
    nb.save(nb.Nifti1Image(tsnr, img.get_affine()), outname)

    return os.path.abspath(outname)
