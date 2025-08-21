import os 
from glob import glob

import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio

from skimage.transform import resize
from joblib import Parallel, delayed

from brainbuilder.utils.utils import resample_to_resolution

def create_init_dataframe():
    init_csv = 'init_dataframe.csv'
    
    if not os.path.exists(init_csv) or True :
        macaque_image_list = glob('/home/tfunck/projects/nnUNet_validation/receptor_annotations_macaque/to_predict/*nii.gz') 
        human_image_list = glob('/home/tfunck/projects/nnUNet_validation/receptor_annotations_human/to_predict/*nii.gz')

        image_list = macaque_image_list + human_image_list
        
        get_basename = lambda x : x.split('/')[-1].split('_')[0].replace('.nii.gz','').replace('.png', '').replace('.TIF','')

        basenames = [ get_basename(x) for x in image_list]

        png_image_list = glob('image/*')

        ymha_image_list = [ x for x in png_image_list if get_basename(x) not in basenames ]
        print(ymha_image_list)

        n_original = len(macaque_image_list) + len(human_image_list)

        n_ymha = len(ymha_image_list)

        source_list = ['original'] * n_original + ['ymha'] * n_ymha

        ymha_basenames = [ get_basename(x) for x in ymha_image_list]

        basenames += ymha_basenames

        image_list += ymha_image_list

        df_list = []

        for image, basename, source in zip(image_list, basenames, source_list):
            
            species = 'human'

            if '11530' in basename or '11539' in basename or '11540' in basename:
                species = 'macaque'

            row = pd.DataFrame( {
                'image': [image],
                'basename': [basename],
                'rotate': [0],  # default rotation
                'flip_0': [False],  # default flip
                'flip_1': [False],  # default flip
                'source': [source],
                'species': [species],
                'modality' : ['autoradiography']
            })

            df_list.append(row)

        df = pd.concat(df_list, ignore_index=True)
        
        df.to_csv(init_csv, index=False)
    else:
        df = pd.read_csv(init_csv)
    
    return df


def set_labels(df: pd.DataFrame):

    label_list = glob('/home/tfunck/projects/nnUNet_validation/receptor_annotations_macaque/label/*nii.gz') 
    label_list += glob('/home/tfunck/projects/nnUNet_validation/receptor_annotations_human/label/*nii.gz')
    label_list += glob('label/*')

    df['label'] = [''] * len(df)

    for i, row in df.iterrows():
        
        basename = row['basename']
        
        temp_label_list = [ x for x in label_list if basename in x]
        
        if len(temp_label_list) == 0:
            print(f'No label found for {basename}')
            continue
        elif len(temp_label_list) > 1:
            print(f'Multiple labels found for {basename}, using the first one.')

        label = temp_label_list[0]

        df.at[i, 'label'] = label
    
    df.sort_values(by='basename', inplace=True)

    return df

def get_raw_path(df:pd.DataFrame, input_dir:str='/data/receptor/human/preprocessed/', target_resolution:str = 0.2 ):

    raw_df_csv = 'raw_dataframe.csv'
    if not os.path.exists(raw_df_csv) or True:
        
        # find raw basename
        df['raw_image'] = [''] * len(df)
        for i, row in df.iterrows():
            raw_path_str = f'{input_dir}/{row["basename"]}*nii.gz'

            raw_path_list = glob(raw_path_str)

            if len(raw_path_list) == 0:
                print(f'No raw image found for {row["basename"]}')
                continue
            elif len(raw_path_list) > 1:
                print(f'Multiple raw images found for {row["basename"]}, using the first one.')

            df.at[i, 'raw_image'] = raw_path_list[0]
    
        df.sort_values(by='basename', inplace=True)
        
        df.to_csv(raw_df_csv, index=False)
    else:
        df = pd.read_csv(raw_df_csv)


    return df   

def apply_transforms(df: pd.DataFrame, image_rsl_dir:str = 'image_rsl/', label_rsl_dir:str = 'label_rsl/', qc_dir:str='qc/', target_resolution:str = 0.2 ):
    # Apply transformations to the dataframe

    os.makedirs(image_rsl_dir, exist_ok=True)
    os.makedirs(label_rsl_dir, exist_ok=True)
    os.makedirs(qc_dir, exist_ok=True)
    
    df['image_rsl'] = df['basename'].apply(lambda x: image_rsl_dir + '/' + os.path.basename(x) + '_0000.nii.gz')
    df['label_rsl'] = df['basename'].apply(lambda x: label_rsl_dir + '/' + os.path.basename(x) + '_label.nii.gz')

    def process_row(row):

        raw_img_hd = nib.load(row['raw_image'])

        if '.png' in row['image']:
            # Load image as a PNG file
            ar = iio.imread(row['image'])
        else :
            img_hd = nib.load(row['image'])
            ar = np.squeeze( img_hd.get_fdata() )

        ar_raw = np.squeeze( raw_img_hd.get_fdata() )

        dim_scale = np.mean( np.array(ar_raw.shape) / np.array(ar.shape))

        effective_resolution = np.array(raw_img_hd.header.get_zooms()) * dim_scale

        rotate = int(row['rotate'])

        if rotate != 0:
            ar = np.rot90(ar, k=rotate)
        if row['flip_0']:
            ar = np.flip(ar, axis=0)
        if row['flip_1']:
            ar = np.flip(ar, axis=1)
        # Save the processed image back or return it

        print(f'Processing {row["basename"]} with shape {ar.shape} and rotation {row["rotate"]}, flip_0: {row["flip_0"]}, flip_1: {row["flip_1"]}')
        
        affine = np.eye(4)
        affine[0,0] = effective_resolution[0]
        affine[1,1] = effective_resolution[1]

        # resample 
        resample_to_resolution(ar, target_resolution, affine=affine, output_filename = row['image_rsl'])
        
        img_rsl = nib.load(row['image_rsl'])
        dims = img_rsl.shape
        ar_rsl = img_rsl.get_fdata()

        if '.png' in row['label']:
            label = iio.imread(row['label'])
        else:
            # Load label as a Nifti image
            label_hd = nib.load(row['label'])
            label = np.squeeze(label_hd.get_fdata())

        if len(label.shape) > 2 :
            a = np.unique(label)
            label = np.mean(label, axis=2)

        label_rsl = resample_to_resolution(label, target_resolution, affine=affine, output_filename=row['label_rsl'], order=0).get_fdata()

        label_rsl[label_rsl < 0.9 * np.max(label_rsl) ] = 0
        label_rsl[label_rsl > 0] = 1

        plt.imshow(ar_rsl, cmap='gray')
        plt.contour(label_rsl, colors='r', linewidths=0.5)
        plt.axis('off')
        plt.savefig(os.path.join(qc_dir, f'{row["basename"]}_source-{row["source"]}_qc.png'), bbox_inches='tight', pad_inches=0)
        plt.close()
        
        return row
    
    results = Parallel(n_jobs=-1)(
        delayed(process_row)(row) for i, row in df.iterrows() if row['label'] != '')

    df = pd.DataFrame(results)

    return df


df = create_init_dataframe()

df = get_raw_path(df)
df = set_labels(df)

transform_df = pd.read_csv('tranform_dataframe.csv')

# drop rows with no label
df = df[df['label'] != '']
df = df[df['raw_image'] != '']
df = df[df['image'] != '']

#if transform_df.shape[0] > 0:
#    del df['rotate']
#    del df['flip_0']
#    del df['flip_1']
#    df = pd.merge(df, transform_df, on='basename', how='left')

df = apply_transforms(df)

print(df.groupby('species').size())