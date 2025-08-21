"""Prepare Chimpanzee data for validation."""

import os
import sys
import nibabel as nib
import numpy as np
import imageio.v3 as iio
from glob import glob

chimp_dir = 'chimpanzee_histology'

file_list = glob(f'{chimp_dir}/*png')

for path in file_list :
    nii_path = path.replace('.png', '.nii.gz')
    if not os.path.exists(nii_path):
        print(f'Processing {path}...')
        img = iio.imread(path)
        print(img.shape)
