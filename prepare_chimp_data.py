"""Prepare Chimpanzee data for validation."""

import os
import sys
import nibabel as nib
import numpy as np
import imageio.v3 as iio
from glob import glob
from skimage import restoration

import numpy as np, cv2


chimp_dir = 'chimpanzee'
hist_dir = f'{chimp_dir}/chimpanzee_histology'
nii_dir = f'{chimp_dir}/nii/'

os.makedirs(nii_dir, exist_ok=True)

file_list = glob(f'{hist_dir}/*jpg')

print(f'{chimp_dir}/*png')
print(file_list)

def _make_gaussian_psf(size: int = 7, sigma: float = 0.9) -> np.ndarray:
    size = int(size) | 1  # force odd
    ax = np.linspace(-(size//2), size//2, size)
    X, Y = np.meshgrid(ax, ax)
    psf = np.exp(-(X**2 + Y**2) / (2.0 * sigma * sigma))
    psf /= psf.sum()
    return psf.astype(np.float32)

resolution = 0.163

aff = np.eye(4)
aff[0, 0] = resolution
aff[1, 1] = resolution

for path in file_list :
    nii_path = nii_dir  + os.path.basename(path).split('.')[-2]+ '_0000.nii.gz'
    
    print(f'Processing {path}...')
    img = iio.imread(path)

    img = np.mean(img, axis=-1)

    img = img.max() - img

    img = np.rot90(img, -1)

    # pad image to 1000x1000, keeping the center
    h, w = img.shape
    pad_h = max(0, 1000 - h)
    pad_w = max(0, 1000 - w)
    img = np.pad(img, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
                 mode='constant', constant_values=0)


    img = (img - img.min()) / (img.max() - img.min())

    sigma = 1.
    num_iter = 4
    
    psf = _make_gaussian_psf(sigma=sigma)
    
    nib.Nifti1Image( (img*255).astype(np.uint8), aff).to_filename(nii_path)

    img = restoration.richardson_lucy(img, psf=psf, num_iter=num_iter)

    nib.Nifti1Image( (img*255).astype(np.uint8), aff).to_filename(nii_path.replace('.nii.gz', '_deconv.nii.gz'))