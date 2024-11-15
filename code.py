# -*- coding: utf-8 -*-

!pip install pydicom pylibjpeg[all]

"""# Load and unzip dataset

The **3D-IRCADb-01** database is composed of the 3D CT-scans of 10 women and 10 men with hepatic tumours in 75% of cases. The 20 folders correspond to **20 different patients**, which can be downloaded individually or conjointly. The table below provides information on the image, such as liver size (width, depth, height) or the location of tumours according to Couninaud’s segmentation. It also indicates the major difficulties liver segmentation software may encounter due to the contact with neighbouring organs, an atypical shape or density of the liver, or even artefacts in the image.

Folder description


1.   "**PATIENT_DICOM**": anonymized patient image in DICOM format
2.   "**LABELLED_DICOM**": labelled image corresponding to the various zones of interest segmented in DICOM format
3.   "**MASKS_DICOM**": a new set of sub-folders corresponding to the names of the various segmented zones of interest containing the DICOM image of each mask
4.   "**MESHES_VTK**": all the files corresponding to surface meshes of the various zones of interest in VTK format.
"""

!gdown 1rEO41mg1t_n5atfk-C_LLn9sCaukMiU9

import numpy as np
import matplotlib.pyplot as plt
import random
import math
import seaborn as sns
import re
from scipy.spatial import ConvexHull
from skimage.measure import label
from matplotlib.patches import Polygon,Patch
import matplotlib.colors as mcolors

import warnings # To suppress some warnings

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

import os
import zipfile
from tqdm import tqdm

zip_datasets_folder = os.getcwd()
zip_dataset_name = '3Dircadb1.zip'
zip_dataset_path = os.path.join(zip_datasets_folder, zip_dataset_name)
print(f"Zip dataset path: \'{zip_dataset_path}\'")

dataset_name = '3Dircadb1'
dataset_folder = os.path.join(os.getcwd(),'Dataset')
if not os.path.exists(dataset_folder):
  os.makedirs(dataset_folder)
print(f"Dataset folder: \'{dataset_folder}\'")

if os.path.exists(zip_dataset_path):
  print(f"Found \'{zip_dataset_name}\' at path: \'{zip_dataset_path}\'")
  print(f"Unzip \'{zip_dataset_name}\' and store data at path: \'{dataset_folder}\'")
  with zipfile.ZipFile(zip_dataset_path, 'r') as zip_ref:
    #zip_ref.extractall(dataset_folder)
    file_list = zip_ref.namelist()
    for file in tqdm(file_list, desc="Extracting files", unit="file"):
      zip_ref.extract(file, dataset_folder)

  if os.path.exists(os.path.join(dataset_folder, dataset_name)):
    dataset_path = os.path.join(dataset_folder, dataset_name)
    print(f"\nDataset path : \'{dataset_path}\'")
else:
  raise FileNotFoundError(f"Cannot find \'{zip_dataset_name}\' at path: \'{zip_dataset_path}\'")

dataset_path = '/content/Dataset/3Dircadb1'

dataset_name = '3Dircadb1'

os.listdir(dataset_path)

sub_folder_names = ['PATIENT_DICOM', 'LABELLED_DICOM', 'MASKS_DICOM' ,'MESHES_VTK']
for folder in tqdm(os.listdir(dataset_path), desc = "Extracting sub-folders", unit="folder"):
  if folder in ['.DS_Store']:
    continue
  folder_path = os.path.join(dataset_path, folder)
  for sub_folder in sub_folder_names:
    if sub_folder in ['MESHES_VTK']:
      continue
    zip_subfolder_path = os.path.join(folder_path, sub_folder+'.zip')
    with zipfile.ZipFile(zip_subfolder_path, 'r') as zip_ref:
      file_list = zip_ref.namelist()
      for file in file_list:
        zip_ref.extract(file, folder_path)
    os.remove(zip_subfolder_path)

"""# Load CT-Scans in Dicom format using pydicom"""

import pydicom

"""## Loading dicom file function"""

def read_dicom(file_path):
  dicom_obj = pydicom.dcmread(file_path)
  return dicom_obj

def load_dicom_file(sample_numbers: list[int] = [1], sub_folder: str='PATIENT_DICOM', img_nums: list[str]=['image_0']):
  assert min(sample_numbers) >= 1 and  max(sample_numbers) <= 20
  result = []
  if sub_folder in ['MASKS_DICOM']:
    for i , samp in zip(img_nums, sample_numbers):
      path = os.path.join(dataset_path,f'{dataset_name}.{samp}',sub_folder,'liver',i)
      result.append(read_dicom(path))
    return result
  for i , samp in zip(img_nums, sample_numbers):
    path = os.path.join(dataset_path,f'{dataset_name}.{samp}',sub_folder,i)
    result.append(read_dicom(path))
  return result

"""##Explore data in raw DICOM format"""

sample = load_dicom_file()[0]

"""### Show DICOM file metadata"""

sample

"""### Show some raw ct scan images

####Helper code
"""

def show_ct_images(imgs: list, row = 1, col = 1, titles:list[str] = [], suptitle:str = '', fig_size=(7,7),c_map = plt.cm.bone):
  if row * col < len(imgs):
    print("Not enough plots")
    return None
  if len(imgs) > len(titles):

    titles = titles + ['' for _ in range(len(imgs)-len(titles))]
  if row == col and row == 1:
    plt.figure(figsize=fig_size)
    plt.imshow(imgs[0],cmap=c_map)
    plt.title(titles[0])
    plt.axis("off")
    plt.show()
    return None
  fig, axs = plt.subplots(row, col, figsize=fig_size)

  axs = axs.flatten() if isinstance(axs, (list, np.ndarray)) else [axs]

  for i , (img, ax) in enumerate(zip(imgs, axs)):
    ax.imshow(img, cmap=c_map)
    ax.axis("off")
    ax.set_title(titles[i])
  for j in range(len(imgs), row * col):
    axs[j].axis("off")
  plt.suptitle(suptitle, fontsize=20, fontweight='bold')
  plt.tight_layout()
  plt.show()

random.seed(42)
sample_numbers = sorted(random.sample(range(1, 21), 5))
img_nums = [random.choice(os.listdir(os.path.join(dataset_path,f'{dataset_name}.{i}','PATIENT_DICOM'))) for i in sample_numbers]
print(sample_numbers)
print(img_nums)

#Get dicom object
patient_dicom = load_dicom_file(sample_numbers,sub_folder= 'PATIENT_DICOM', img_nums = img_nums)
label_dicom = load_dicom_file(sample_numbers,sub_folder= 'LABELLED_DICOM', img_nums = img_nums)
masked_dicom = load_dicom_file(sample_numbers,sub_folder= 'MASKS_DICOM', img_nums = img_nums)

#Extract images (pixel_array)
patient_imgs = [i.pixel_array for i in patient_dicom]
label_imgs = [i.pixel_array for i in label_dicom]
liver_masks_imgs = [i.pixel_array for i in masked_dicom]
masked_patient = [np.where(mask == 255, image, np.min(image)) for mask, image in zip(liver_masks_imgs, patient_imgs)]
masked_label = [np.where(mask == 255, image, np.min(image)) for mask, image in zip(liver_masks_imgs, label_imgs)]

patient_titles = [f"Patient {samp}-{im.split('_')[1]}" for samp, im in zip(sample_numbers, img_nums)]
label_titles = [f"Labelled {samp}-{im.split('_')[1]}" for samp, im in zip(sample_numbers, img_nums)]
liver_masks_labels = [f"Liver mask {samp}-{im.split('_')[1]}" for samp, im in zip(sample_numbers, img_nums)]
masked_patient_labels = [f"Masked patient {samp}-{im.split('_')[1]}" for samp, im in zip(sample_numbers, img_nums)]
masked_label_labels = [f"Masked label {samp}-{im.split('_')[1]}" for samp, im in zip(sample_numbers, img_nums)]

"""####Raw data visualization"""

show_ct_images(patient_imgs, row = 1 , col = 5, titles = patient_titles , suptitle = "PATIENTS CT SCAN in HU format", fig_size = (15,5))

show_ct_images(label_imgs, row = 1 , col = 5, titles = label_titles, suptitle = "LABELLED CT SCANS (organs are labelled with values from 0 to 255)", fig_size = (15,5))

show_ct_images(liver_masks_imgs, row = 1 , col = 5, titles = label_titles, suptitle = "MASKED LIVER IMAGES (masked organs are marked with value 1 or 255 and the rest of the image is 0)", fig_size = (15,5))

"""## Set up dataframe"""

def extractStr(string:str):
  patient_name = string.replace("^", "_")
  splt = patient_name.split("_")
  return int(splt[1])

from concurrent.futures import ThreadPoolExecutor

def process_ctscan(subfolder_path, item, key):
    path_lst = [(os.path.join(subfolder_path, item, file)) for file in os.listdir(os.path.join(subfolder_path, item))]
    lst = [read_dicom(path) for path in path_lst]
    obj_lst = [{
        "Path": path,
        "PatientName": key,
        "ImageNum": (((path.split('/'))[-1]).split('_'))[-1],
        "ImageSize": obj.pixel_array.shape,
        "RescaleIntercept": obj.RescaleIntercept if 'RescaleIntercept' in obj else None,
        "RescaleSlope": obj.RescaleSlope if 'RescaleSlope' in obj else None,
        "MaxHU": np.max(obj.pixel_array),
        "MinHU": np.min(obj.pixel_array),
        "HURange": np.max(obj.pixel_array) - np.min(obj.pixel_array)
    } for obj, path in zip(lst,path_lst)]
    return obj_lst

def process_mask(path_lst, part, key):
    obj_list = []
    for path in path_lst:
        obj = read_dicom(path)
        mask = (obj.pixel_array / 255.0).astype(float)
        obj_list.append({
            'Path': path,
            'PatientName': key,
            "ImageNum": (((path.split('/'))[-1]).split('_'))[-1],
            'OrganName': part,
            'OrganSize': np.sum(np.array(mask) != 0)
        })
    return obj_list

def process_patient_data(num, dataset_path, dataset_name):
    key = f'Patient_{num + 1}'
    subfolder_path = os.path.join(dataset_path, f'{dataset_name}.{num+1}')

    patient_data = {
        'CTScan': [],
        'OrgansInfo': {}
    }

    # Process CTScans
    patient_data['CTScan'] = process_ctscan(subfolder_path, 'PATIENT_DICOM', key)

    # Process Organs/Masks
    parts = os.listdir(os.path.join(subfolder_path, 'MASKS_DICOM'))
    for part in parts:
        path_lst = [(os.path.join(subfolder_path, 'MASKS_DICOM', part, file)) for file in os.listdir(os.path.join(subfolder_path, 'MASKS_DICOM', part))]
        patient_data['OrgansInfo'][part] = process_mask(path_lst, part, key)

    return key, patient_data

# Main execution with threading
dataset_dict = {}

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(process_patient_data, num, dataset_path, dataset_name) for num in range(20)]
    for future in futures:
        key, patient_data = future.result()
        dataset_dict[key] = patient_data

import pandas as pd
import itertools

patient_df = pd.DataFrame(itertools.chain.from_iterable([(dataset_dict[patient]['CTScan']) for patient in dataset_dict.keys()]))

len(patient_df)

mask_df = pd.DataFrame(itertools.chain.from_iterable([dataset_dict[patient]['OrgansInfo'][organ] for patient in dataset_dict.keys() for organ in dataset_dict[patient]['OrgansInfo']]))

len(mask_df)

mask_df.head(5)

patient_df.to_csv('patient_df.csv', index=False)  # 'index=False' if you don't want to store the index
mask_df.to_csv('mask_df.csv', index=False)  # 'index=False' if you don't want to store the index



"""#Transform from HU to GrayScale and apply windowing for better visualization

##Transform to HU
"""

def transform_to_hu(medical_image, image):
  intercept = medical_image.RescaleIntercept
  slope = medical_image.RescaleSlope
  hu_image = image * slope + intercept
  return hu_image

"""##Window operation"""

def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max

    return window_image

"""## Normalize image"""

def normalize_img(img):
  i_max = np.max(img)
  i_min = np.min(img)
  if i_max == i_min :
    return img
  return (img-i_min)/(i_max-i_min) # range [0,1]

"""##Preprocess a CT-Scan image to GrayScale using windowing"""

def preprocess(dicom_obj, window: tuple|None = None):
  image = dicom_obj.pixel_array
  hu_image = transform_to_hu(dicom_obj, image)
  if window is not None:
    transform_image = window_image(hu_image, window[0], window[1])
  else:
    transform_image = hu_image

  norm_image = normalize_img(transform_image) * 255.0
  #print(np.max(norm_image),np.min(norm_image))
  result = norm_image.astype(int)
  #print(np.max(result),np.min(result))
  return result

"""##Show tumor and liver"""

def making_samples(mask_df, patient_df, state=42):
  def match_str(string):
      # Define the regular expression pattern
      pattern = r'^livertumor(s|(0[1-9]|1[0-9]|20))?$'

      # Use re.match() to check if the string matches the pattern
      return bool(re.match(pattern, string))

  # Sample liver samples with unique PatientName values
  liver_samples = mask_df[(mask_df['OrganName'] == 'liver') & (mask_df['OrganSize'] > 0)]

  # Group by PatientName and sample one per group, then take the first 4
  random.seed(state)

  liver_samples = liver_samples.groupby('PatientName').apply(lambda x: x.sample(1, random_state=state)).reset_index(drop=True)

  # If there are fewer than 4 unique PatientNames, limit the result to available samples
  if len(liver_samples) > 4:
      liver_samples = liver_samples.sample(4, random_state=state)

  # Get unique combinations of PatientName and ImageNum from liver_samples
  liver_combinations = liver_samples[['PatientName', 'ImageNum']].drop_duplicates()

  # Filter tumor samples based on matching combinations and other conditions
  tumor_samples = mask_df[
      (mask_df[['PatientName', 'ImageNum']].apply(tuple, axis=1).isin(liver_combinations.apply(tuple, axis=1)))
  ]

  tumor_samples=tumor_samples[(tumor_samples['OrganName'].apply(match_str)) & (tumor_samples['OrganSize'] > 0)]

  tumor_samples=tumor_samples.merge(liver_samples[['PatientName', 'ImageNum', 'OrganSize','Path']],on=['PatientName', 'ImageNum'], how='left')

  tumor_samples = tumor_samples.rename(columns={'OrganSize_x': 'TumorSize', 'OrganSize_y': 'LiverSize', 'Path_x': 'TumorPath', 'Path_y': 'LiverPath'})

  tumor_samples=tumor_samples.merge(patient_df[['PatientName', 'ImageNum','Path']], on=['PatientName', 'ImageNum'], how='left')

  return tumor_samples

tumor_samples = making_samples(mask_df, patient_df,140)

def visualizeTumor(df, window= (50,120), fig_size=(7,7)):
  tumor_mask = [preprocess(read_dicom(path)) for path in tumor_samples['TumorPath']]
  liver_mask = [preprocess(read_dicom(path)) for path in tumor_samples['LiverPath']]
  patient_img = [preprocess(read_dicom(path),window) for path in tumor_samples['Path']]

  num_pairs = len(patient_img)

  fig, axs = plt.subplots(num_pairs, 4, figsize=fig_size)

  axs = axs.flatten() if isinstance(axs, (list, np.ndarray)) else [axs]

  for i in range(num_pairs):
    axs[4*i].imshow(patient_img[i], cmap='gray')  # Show the image on the first column
    axs[4*i].axis('off')  # Turn off axis for the image

    axs[4*i + 1].imshow(liver_mask[i], cmap ='gray')  # Show the label image on the second column
    axs[4*i + 1].axis('off')  # Turn off axis for the label image

    axs[4*i + 2].imshow(tumor_mask[i], cmap ='gray')  # Show the label image on the second column
    axs[4*i + 2].axis('off')  # Turn off axis for the label image

    color_img = np.stack([patient_img[i], patient_img[i], patient_img[i]], axis=-1)
    color_img[(liver_mask[i] == 255)] = [0, 0, 255]
    color_img[(tumor_mask[i] == 255)] = [255, 0, 0]
    axs[4*i + 3].imshow(color_img)
    axs[4*i + 3].axis('off')

  plt.tight_layout()
  plt.show()

  #return patient_img, liver_mask, tumor_mask

visualizeTumor(tumor_samples)

"""##Visualization"""

patient_df

"""###Abdomen Window - (W:400, L:40)

The abdomen window is used to evaluate the abdominal cavity and its contents. A window level close to fluid or soft tissue is used with a medium-sized window that gives a moderate amount of contrast.
"""

window = (40,400)

samples = patient_df.groupby('PatientName').apply(lambda x: x.sample(1, random_state=42)).reset_index(drop=True)
images = [preprocess(read_dicom(path),window) for path in samples['Path']]

show_ct_images(images, row = 4 , col = 5, suptitle = "Abdomen Window of all patients", fig_size = (20,20))

"""###Soft tissue window - (W:350, L:50)

As the name suggests, this window is used to evaluate soft tissues. The window level is set at the density of soft tissues (50 HU) and a moderate-sized window is used to give a balance between contrast and resolution. Best for reviewing solid organs and vasculature, we can see the heart, liver, aorta, and major vessels.
"""

window = (50,350)

samples = patient_df.groupby('PatientName').apply(lambda x: x.sample(1, random_state=42)).reset_index(drop=True)
images = [preprocess(read_dicom(path),window) for path in samples['Path']]

show_ct_images(images, row = 4 , col = 5, suptitle = "Soft Tissue Window of all patients", fig_size = (20,20))

"""###Liver Window - (W:160, L:60)

This window is similar to the abdominal window however utilizes a narrower window to increase the contrast in the liver parenchyma (in order to make finding hepatic lesions a bit easier).
"""

window = (60,160)

samples = patient_df.groupby('PatientName').apply(lambda x: x.sample(1, random_state=42)).reset_index(drop=True)
images = [preprocess(read_dicom(path),window) for path in samples['Path']]

show_ct_images(images, row = 4 , col = 5, suptitle = "Liver Window of all patients", fig_size = (20,20))

"""###Bone Window - (W:2000, L:500) or (W:3000, L:1000)

As the name suggests, a bone window is useful for viewing the bones. A high window level near the density of bone (given its density the level is high) is used with a wide window to give a good resolution.
"""

window = (500,2000)

samples = patient_df.groupby('PatientName').apply(lambda x: x.sample(1, random_state=42)).reset_index(drop=True)
images = [preprocess(read_dicom(path),window) for path in samples['Path']]

show_ct_images(images, row = 4 , col = 5, suptitle = "Bone Window of all patients", fig_size = (20,20))

"""###Lung Window - (W:1600, L:-600) or (W:1500, L:-500)

This window is used to evaluate the lungs. A high window level near the density of lung tissue (given its low density the level is low) and is used with a wide window to give good resolution and to visualize a wide variety of densities in the chest such as the lung parenchyma as well as adjacent blood vessels.
"""

window = (-600,1600)

samples = patient_df.groupby('PatientName').apply(lambda x: x.sample(1, random_state=42)).reset_index(drop=True)
images = [preprocess(read_dicom(path),window) for path in samples['Path']]

show_ct_images(images, row = 4 , col = 5, suptitle = "Lung Window of all patients", fig_size = (20,20))

"""#Organ Labels Visualization

In this part, we are going to focus on showing what data is labelled in each CT scans
"""

def plot_scan_and_labels(masks,labels,image,title):
  plt.figure(figsize=(12,10))
  plt.imshow(image, cmap='gray')
  plt.title(title)
  colors = plt.cm.get_cmap('tab20', len(masks))
  legend_patch = []
  for i in range(len(masks)):
      # Step 1: Label the distinct regions in the mask
    # Step 1: Label the distinct regions in the mask
    labeled_mask, num_labels = label(masks[i], return_num=True)
    # Step 3: Loop through each labeled region and find its convex hull
    for region_id in range(1, num_labels + 1):  # Labels start at 1
        region_points = np.column_stack(np.where(labeled_mask == region_id))  # Get points for each region

        if len(region_points) > 2:  # ConvexHull requires at least 3 points
            hull = ConvexHull(region_points)

            # Extract hull vertices
            hull_vertices = region_points[hull.vertices]
            # Plot the convex hull as a Polygon (with only edges, no fill)
            polygon = Polygon(np.fliplr(hull_vertices), facecolor='none', edgecolor=colors(i), linewidth=2)
            plt.gca().add_patch(polygon)
    legend_patch.append(Patch(facecolor='none', edgecolor=colors(i), label=labels[i], linewidth=2))
    # Step 5: Create a custom legend ßfor the convex hull
  plt.legend(handles=legend_patch, loc='upper right', title='Legend')

    # Show the plot with all convex hulls

sample_patients = patient_df.sample(n=10,random_state=42)

sample_patients

sample_patients['Path']

images = [preprocess(read_dicom(path),(40,400)) for path in sample_patients['Path']]

titles = list(sample_patients['PatientName']+', Image number '+sample_patients['ImageNum'])

titles[0]

masks_dfs = [pd.DataFrame([x]).merge(mask_df[['PatientName', 'ImageNum','Path','OrganSize','OrganName']],on=['PatientName', 'ImageNum'],how='left') for i,x in sample_patients.iterrows()]
masks_dfs = [x[x['OrganSize']>0] for x in masks_dfs]

masks = [[preprocess(read_dicom(path)) for path in x['Path_y']] for x in masks_dfs]
labels = [list(x['OrganName']) for x in masks_dfs]



for i in range(len(images)):
  plot_scan_and_labels(masks[i],labels[i],images[i],titles[i])

"""# Statistics

##Overview

###Number of images per patient
"""

# Visualize the number of images of each patient
patient_counts = patient_df.groupby('PatientName').size()

patient_counts = patient_df[['PatientName']].drop_duplicates().set_index('PatientName').join(patient_counts.rename('count'))

plt.figure(figsize=(12, 5))

bars = patient_counts['count'].plot(kind='bar', color='skyblue', width=0.5)

plt.xlabel('Patient Name')
plt.ylabel('Number of Images')
plt.title('Number of Images per Patient')

max_value = patient_counts['count'].max()
min_value = patient_counts['count'].min()

textstr = f'Max: {max_value}\nMin: {min_value}'

plt.text(0.80, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

for bar in plt.gca().patches:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', va='bottom')

plt.show()

"""###Number of folder (organ segments) per patient"""

# Visualize the number of images of each patient
patient_organs = mask_df.groupby('PatientName')['OrganName'].nunique().reset_index()

patient_organs.columns = ['PatientName', 'OrgansCount']

patient_organs['PatientNum'] = patient_organs['PatientName'].str.extract(r'(\d+)').astype(int)
patient_organs = patient_organs.sort_values('PatientNum').drop('PatientNum', axis=1)

plt.figure(figsize=(12, 5))

bars = plt.bar(patient_organs['PatientName'], patient_organs['OrgansCount'], color = 'skyblue', width = 0.5)

plt.xlabel('Patient Name')
plt.ylabel('Number of Folders')
plt.title('Number of Folders per Patient')

plt.xticks(rotation=90)
plt.yticks(np.arange(0, patient_organs['OrgansCount'].max() + 2, 2))

max_value = patient_organs['OrgansCount'].max()
min_value = patient_organs['OrgansCount'].min()

textstr = f'Max: {max_value}\nMin: {min_value}'

plt.text(0.80, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', va='bottom')

# max_value = patient_counts['count'].max()
# min_value = patient_counts['count'].min()

# textstr = f'Max: {max_value}\nMin: {min_value}'

# plt.text(0.80, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
#          verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

plt.show()

"""###HU range distribution of all images"""

plt.figure(figsize=(7, 5))

sns.histplot(patient_df['HURange'], kde=False, color='blue', bins=20, label='HURange')

plt.xlabel('HU Ranges')
plt.ylabel('Count')
plt.title('HU Range Distribution')


plt.show()

"""##Detail statistics

###Average number of pixels for each label (across all slices) for each Patient
"""

def custom_mean(group):
    # Filter the series to include only values > 0
    series = group['OrganSize']
    positive_values = series[series > 0]

    total_sum = positive_values.sum()  # Sum of values greater than 0
    count = positive_values.count()    # Count of values greater than 0

    # Return the mean if count is greater than 0, else return NaN
    #return total_sum / count if count > 0 else 0.0

    return pd.Series({
        'AvgOrganSize': total_sum / count if count > 0 else 0.0,
    })


averages = mask_df.groupby(['PatientName', 'OrganName']).apply(custom_mean,include_groups=False).reset_index()
# Create a column chart for each Patient

# Extract numerical part of PatientName to create a sortable column
averages['PatientNum'] = averages['PatientName'].str.extract(r'_(\d+)').astype(int)

# Sort by the extracted PatientNum, then by ImageNum, and finally by OrganName
averages = averages.sort_values(by=['PatientNum', 'OrganName'])

# Drop the PatientNum column if it's no longer needed
averages = averages.drop(columns='PatientNum')

patients = averages['PatientName'].unique()

# Set the size of the plots
plt.figure(figsize=(50, 50))

# Number of rows and columns
rows = 4
cols = 5
plt.suptitle('Average Organ Size (in pixels) for each patient', fontsize=20, fontweight='bold')

for i, patient in enumerate(patients):
    plt.subplot(rows, cols, i + 1,)  # Create a subplot for each patient
    patient_data = averages[averages['PatientName'] == patient]

    # Create a bar plot for average OrganSize
    bars = sns.barplot(x='OrganName', y='AvgOrganSize', data=patient_data, palette='viridis', hue='AvgOrganSize', legend=False,width=0.2)

    plt.title(f'Average Organ Size for {patient}')
    plt.xlabel('Organ Name')
    plt.ylabel('Average Organ Size')

    # Rotate x-axis labels
    plt.xticks(rotation=90)

    # Add values on top of bars
    for bar in bars.patches:
        height = bar.get_height()
        if height > 0:  # Only display text for positive heights
            bars.text(bar.get_x() + bar.get_width() / 2., height,
                      f'{height:.2f}', ha='center', va='bottom', fontsize=6)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to prevent overlap with suptitle
plt.show()

"""###Liver size distribution across all images

We want to see the size of the liver across all images to get a general view about the distribution of liver size in CT scans
"""

plt.figure(figsize=(7, 5))

sns.histplot(mask_df[(mask_df['OrganName']=='liver')]['OrganSize'], kde=False, color='blue', bins=50, label='OrganSize')

plt.xlabel('LiverSize (in pixels)')
plt.ylabel('Count')
plt.title('Liver size Distribution')


plt.show()

"""###Other organs size distribution

Similarly we want to see the size statistics for other organs as well to get a general view about their size distribution
"""

mask_df['OrganName'].unique()

organs = [ 'rightkidney', 'leftlung',
       'portalvein', 'skin',
         'liver',
       'liverkyst', 'spleen', 'rightlung', 'leftkidney', 'venoussystem',
       'artery', 'bone', 'gallbladder', 'livertumor', 'venacava',
       'biliarysystem', 'Stones', 'metal', 'kidneys', 'stomach', 'lungs',
       'rightsurrenalgland', 'pancreas', 'rightsurretumor',
       'leftsurretumor', 'leftsurrenalgland', 'tumor', 'liverkyste',
       'portalvein1', 'metastasectomie',
       'livercyst', 'surrenalgland', 'smallintestin',
       'bladder', 'colon', 'uterus', 'heart']

len(organs)

def match_organ(organ):
  def inner(x):
    if(organ=='livertumor'):
      return organ in x
    else:
      return organ == x
  return inner

for organ in organs:
  data = mask_df[mask_df['OrganName'].apply(match_organ(organ))]['OrganSize']
  plt.figure(figsize=(7, 5))

  sns.histplot(data, kde=False, color='blue', bins=50, label='OrganSize')

  plt.xlabel(f'{organ}Size (in pixels)')
  plt.ylabel('Count')
  plt.title(f'{organ} size Distribution')


  plt.show()

"""###Organ location heatmap

We want to know to overall placement of different organs in these CT scans
"""

for organ in organs:
    paths = mask_df[mask_df['OrganName'].apply(match_organ(organ))]['Path'].tolist()

    # Initialize heat matrix with zeros (assuming all arrays have the same shape)
    heat_matrix = np.zeros(read_dicom(paths[0]).pixel_array.shape, dtype=np.int32)

    # Accumulate values into the heat_matrix
    for path in paths:
        dicom_pixel_array = np.where(read_dicom(path).pixel_array != 0, 1, 0)
        heat_matrix += dicom_pixel_array

    # Plot heatmap
    plt.imshow(heat_matrix, cmap='hot', interpolation='nearest')
    plt.title(f'{organ} Heat Map')
    plt.colorbar()
    plt.show()

"""####Liver vs Liver Tumor heatmap"""

import numpy as np
import matplotlib.pyplot as plt

# Create a figure with two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

for i, organ in enumerate(['liver', 'livertumor']):
    paths = mask_df[mask_df['OrganName'].apply(match_organ(organ))]['Path'].tolist()

    # Initialize heat matrix with zeros (assuming all arrays have the same shape)
    heat_matrix = np.zeros(read_dicom(paths[0]).pixel_array.shape, dtype=np.int32)

    # Accumulate values into the heat_matrix
    for path in paths:
        dicom_pixel_array = np.where(read_dicom(path).pixel_array != 0, 1, 0)
        heat_matrix += dicom_pixel_array

    # Plot the heatmap in the corresponding subplot
    ax = axes[i]  # Select the correct subplot (axes[0] for 'liver', axes[1] for 'livertumor')
    img = ax.imshow(heat_matrix, cmap='hot', interpolation='nearest')
    ax.set_title(f'{organ.capitalize()} Heat Map')

    # Add color bar to each subplot
    fig.colorbar(img, ax=ax)

# Adjust layout
plt.tight_layout()
plt.show()

"""As we can see, the heat map of liver tumor almost mimic the shape of the liver heat map

###Liver Tumor ratio statistics

####Create a dataframe for liver tumor ratio
"""

liver_df = mask_df[mask_df['OrganName'] == 'liver'].drop(columns='Path')


def match_livertumor(string):
    # Define the regular expression pattern
    pattern = r'^livertumors?([0-9])*$'

    # Use re.match() to check if the string matches the pattern
    return bool(re.match(pattern, string))

liver_tumor_df = mask_df[mask_df['OrganName'].apply(match_livertumor)].drop(columns='Path')
liver_tumor_df = liver_tumor_df.merge(liver_df[['PatientName', 'ImageNum', 'OrganSize']],
                                 on=['PatientName', 'ImageNum'],
                                 how='left')
# Rename the 'OrganSize' column from liver_df to 'LiverSize' in the merged DataFrame
liver_tumor_df = liver_tumor_df.rename(columns={'OrganSize_x': 'TumorSize', 'OrganSize_y': 'LiverSize'})
# Extract numerical part of PatientName to create a sortable column
liver_tumor_df['PatientNum'] = liver_tumor_df['PatientName'].str.extract(r'_(\d+)').astype(int)
# Sort by the extracted PatientNum, then by ImageNum, and finally by OrganName
liver_tumor_df = liver_tumor_df.sort_values(by=['PatientNum', 'ImageNum', 'OrganName'])
# Drop the PatientNum column if it's no longer needed
liver_tumor_df = liver_tumor_df.drop(columns='PatientNum').reset_index()

# Get the indices of rows where LiverSize is equal to 0
indices_to_drop = liver_tumor_df[(liver_tumor_df['LiverSize'] == 0) | (liver_tumor_df['TumorSize'] == 0)].index
# Drop those rows
liver_tumor_df = liver_tumor_df.drop(indices_to_drop)

liver_tumor_df['TumorRatio'] = liver_tumor_df['TumorSize'] / liver_tumor_df['LiverSize']

"""###Tumor ratio across all slices

We also want see the ratio distribution of liver tumor to detect anything unusual
"""

plt.figure(figsize=(7, 5))

sns.histplot(liver_tumor_df['TumorRatio'], kde=False, color='blue', bins=50, label='TumorRatio')

plt.xlabel('Tumor Ratio (#pixels-tumors/#pixels-liver)')
plt.ylabel('Count')
plt.title('Tumor Ratio Distribution')


plt.show()

"""As we can see, there are some unusual points in our graph, some tumors have ratio 1 meaning it has size equal of the whole liver, we will investigate further this point

We will select ratio 0.8 for finding suspicious data points
"""

liver_tumor_df

x = liver_tumor_df[liver_tumor_df['TumorRatio']>=0.8].merge(patient_df[['PatientName', 'ImageNum','Path']],
                                 on=['PatientName', 'ImageNum'],
                                 how='left')

len(x)

liver_mask_df = mask_df[(mask_df['OrganName'].apply(match_organ('liver')))]
livertumor_mask_df = mask_df[(mask_df['OrganName'].apply(match_organ('livertumor')))]

x = x.merge(liver_mask_df[['PatientName', 'ImageNum','Path']],
                                 on=['PatientName', 'ImageNum'],
                                 how='left')
x = x.rename(columns={"Path_x": "PatientPath", "Path_y": "LiverMaskPath"})
x = x.merge(livertumor_mask_df[['PatientName', 'ImageNum','Path']],
                                 on=['PatientName', 'ImageNum'],
                                 how='left')
x = x.rename(columns={"Path": "LiverTumorMask"})
x.drop_duplicates(subset=['PatientName','ImageNum'])

images = [(preprocess(read_dicom(path1),(60,160)),preprocess(read_dicom(path2)),preprocess(read_dicom(path3))) for path1,path2,path3 in zip(x['PatientPath'],x['LiverMaskPath'],x['LiverTumorMask'])]
import matplotlib.pyplot as plt

# Assuming 'preprocess' is a function that processes the DICOM images
images = [(preprocess(read_dicom(path1), (60,160)), preprocess(read_dicom(path2)), preprocess(read_dicom(path3)))
          for path1, path2, path3 in zip(x['PatientPath'], x['LiverMaskPath'], x['LiverTumorMask'])]

# Loop through each set of triple arrays (Patient, Liver Mask, Liver Tumor Mask)
for patient_img, liver_mask_img, liver_tumor_img in images:
    # Create a figure with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # Adjust figsize if needed

    # Display the patient image
    axes[0].imshow(patient_img, cmap='gray')  # Use cmap='gray' for grayscale images
    axes[0].set_title('Patient Image')
    axes[0].axis('off')  # Hide axes

    # Display the liver mask image
    axes[1].imshow(liver_mask_img, cmap='gray')
    axes[1].set_title('Liver Mask')
    axes[1].axis('off')

    # Display the liver tumor mask image
    axes[2].imshow(liver_tumor_img, cmap='gray')
    axes[2].set_title('Liver Tumor Mask')
    axes[2].axis('off')

    # Show the plot with all three images on a single line
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

"""We might need more time to investigate further this outlier case but looking at the display we think that the labeling might be true that the tumor has the size of the liver in the CT scan"""

averages_liver = liver_tumor_df.groupby(['PatientName', 'OrganName'])['TumorRatio'].mean().reset_index()
# Create a column chart for each Patient

averages_liver['PatientNum'] = averages_liver['PatientName'].str.extract(r'(\d+)').astype(int)
averages_liver = averages_liver.sort_values('PatientNum').drop('PatientNum', axis=1)

patients = averages_liver['PatientName'].unique()

# Set the size of the plots
plt.figure(figsize=(20, 20))

# Number of rows and columns
rows = 4
cols = 5

for i, patient in enumerate(patients):
    plt.subplot(rows, cols, i + 1)  # Create a subplot for each patient
    patient_data = averages_liver[averages_liver['PatientName'] == patient]

    # Create a bar plot for average OrganSize
    bars = sns.barplot(x='OrganName', y='TumorRatio', data=patient_data, palette='viridis', hue='OrganName', legend=False, width=0.4)

    plt.title(f'Average Tumour Size for {patient}')
    plt.xlabel('Tumor')
    plt.ylabel('Average Tumour Size')

    # Rotate x-axis labels
    plt.xticks(rotation=90)

    plt.yticks(np.arange(0, 1.2, 0.2))


    # Add values on top of bars
    for bar in bars.patches:
        height = bar.get_height()
        if height > 0:  # Only display text for positive heights
            bars.text(bar.get_x() + bar.get_width() / 2., height,
                      f'{height:.3f}', ha='center', va='bottom', fontsize=9)
plt.suptitle('Average Tumor Ratio (#pixels-tumors/#pixels-liver) for each patient across all slices', fontsize=20, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to prevent overlap with suptitle
plt.show()

averages_liver = liver_tumor_df.groupby(['PatientName'])['TumorRatio'].mean().reset_index()
# Create a column chart for each Patient

averages_liver['PatientNum'] = averages_liver['PatientName'].str.extract(r'(\d+)').astype(int)
averages_liver = averages_liver.sort_values('PatientNum').drop('PatientNum', axis=1)

patients = averages_liver['PatientName'].unique()

plt.figure(figsize=(12, 5))

bars = plt.bar(averages_liver['PatientName'], averages_liver['TumorRatio'], color = 'skyblue', width = 0.5)

plt.xlabel('Patient Name')
plt.ylabel('Mean Tumor Ratio')
plt.title('Mean Tumor Ratio Patient')

plt.xticks(rotation=90)
plt.yticks(np.arange(0, 1.2, 0.2))


max_value = averages_liver['TumorRatio'].max()
min_value = averages_liver['TumorRatio'].min()

textstr = f'Max: {max_value:.3f}\nMin: {min_value:.3f}'

plt.text(0.80, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

for bar in bars:
    yval = bar.get_height()

    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, f'{yval:.3f}', ha='center', va='bottom')  # Format to 3 decimal places

plt.show()