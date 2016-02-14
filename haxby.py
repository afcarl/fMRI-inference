from nilearn import datasets
haxby_dataset = datasets.fetch_haxby()

# print basic information on the dataset
print('First subject anatomical nifti image (3D) is at: %s' %
      haxby_dataset.anat[0])
print('First subject functional nifti images (4D) are at: %s' %
      haxby_dataset.func[0])  # 4D data

# Load the behavioral labels
import numpy as np
# Load target information as string and give a numerical identifier to each
labels = np.recfromcsv(haxby_dataset.session_target[0], delimiter=" ")

# scikit-learn >= 0.14 supports text labels. You can replace this line by:
target = labels['labels']
# _, target = np.unique(labels['labels'], return_inverse=True)

# Keep only data corresponding to faces or cats
condition_mask = np.logical_or(labels['labels'] == b'face',
                               labels['labels'] == b'cat')
target = target[condition_mask]

from nilearn.input_data import NiftiMasker
mask_filename = haxby_dataset.mask_vt[0]
# For decoding, standardizing is often very important
nifti_masker = NiftiMasker(mask_img=mask_filename, standardize=True)

func_filename = haxby_dataset.func[0]
# We give the nifti_masker a filename and retrieve a 2D array ready
# for machine learning with scikit-learn
fmri_masked = nifti_masker.fit_transform(func_filename)

# Restrict the classification to the face vs cat discrimination
fmri_masked = fmri_masked[condition_mask]
