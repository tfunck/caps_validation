import os 
from glob import glob

import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import resize
from joblib import Parallel, delayed

from brainbuilder.utils.utils import resample_to_resolution

def create_init_dataframe():
    init_csv = 'init_dataframe.csv'
    
    if not os.path.exists(init_csv):
        image_list = glob('image/*.png')
        basenames = [x.split('/')[-1].split('_')[0].replace('.png', '') for x in image_list]

        label_list = glob('label/*.png')

        df_list = []

        for image, basename in zip(image_list, basenames):
            # find corresponding label
            temp_label_list = [ x for x in label_list if basename in x]
            
            if len(temp_label_list) == 0:
                print(f'No label found for {basename}')
                continue
            elif len(temp_label_list) > 1:
                print(f'Multiple labels found for {basename}, using the first one.')
            label = temp_label_list[0]
            
            row = pd.DataFrame( {
                'image_png': [image],
                'label': [label],
                'basename': [basename],
                'rotate': [0],  # default rotation
                'flip_0': [False],  # default flip
                'flip_1': [False],  # default flip
            })

            df_list.append(row)

        df = pd.concat(df_list, ignore_index=True)
        
        df.to_csv(init_csv, index=False)
    else:
        df = pd.read_csv(init_csv)
    
    return df


def get_raw_path(df:pd.DataFrame, input_dir:str='raw_path/', target_resolution:str = 0.2 ):

    raw_df_csv = 'raw_dataframe.csv'
    if not os.path.exists(raw_df_csv):
        
        # find raw basename
        df['raw_images'] = [''] * len(df)
        for i, row in df.iterrows():
            raw_path_list = glob(f'{input_dir}/{row["basename"]}_*nii.gz')
            if len(raw_path_list) == 0:
                print(f'No raw image found for {row["basename"]}')
                continue
            elif len(raw_path_list) > 1:
                print(f'Multiple raw images found for {row["basename"]}, using the first one.')
            raw_path = raw_path_list[0]
            df.at[i, 'raw_images'] = raw_path
    
        df.to_csv(raw_df_csv, index=False)
    else:
        df = pd.read_csv(raw_df_csv)

    return df   

def apply_transforms(df: pd.DataFrame, image_rsl_dir:str = 'image_rsl/', label_rsl_dir:str = 'label_rsl/', qc_dir:str='qc/', target_resolution:str = 0.2 ):
    # Apply transformations to the dataframe
    
    df['image_rsl'] = df['basename'].apply(lambda x: image_rsl_dir + '/' + os.path.basename(x).replace('.nii.gz', '_0000.nii.gz'))
    df['label_rsl'] = df['basename'].apply(lambda x: label_rsl_dir + '/' + os.path.basename(x).replace('.nii.gz', '_label.nii.gz'))

    def process_row(row):
        img = nib.load(row['raw_path'])
        ar = img.get_fdata()
        if row['rotate'] != 0:
            ar = np.rot90(ar, k=row['rotate'])
        if row['flip_0']:
            ar = np.flip(ar, axis=0)
        if row['flip_1']:
            ar = np.flip(ar, axis=1)
        # Save the processed image back or return it
        
        # resample 
        resample_to_resolution(ar, target_resolution, output_path=row['image_rsl'])
        
        img_rsl = nib.load(row['image_rsl'])
        dims = img_rsl.shape
        ar_rsl = img_rsl.get_fdata()

        label = nib.load(row['label']).get_fdata()
        
        label_rsl = resize(label, dims, order=0, mode='constant', cval=0, anti_aliasing=False)
        
        nib.save(nib.Nifti1Image(label_rsl, img_rsl.affine), row['label_rsl'])
        
        plt.imshow(ar_rsl, cmap='gray')
        plt.contour(label_rsl, colors='r', linewidths=0.5)
        plt.axis('off')
        plt.savefig(os.path.join(qc_dir, f'{row["basename"]}_qc.png'), bbox_inches='tight', pad_inches=0)
        plt.close()
        
        return row
   
    Parallel(n_jobs=-1)(delayed(process_row)(row) for i, row in df.iterrows())
    return df


df = create_init_dataframe()

df = get_raw_path(df)

df = apply_transforms(df)

print(df.head())