# Fetch the data files from Internet
from nilearn import datasets
#from nilearn.image import new_img_like


haxby_dataset = datasets.fetch_haxby(n_subjects=1)

# print basic information on the dataset
print('First subject anatomical nifti image (3D) located is at: %s' %
      haxby_dataset.anat[0])
print('First subject functional nifti image (4D) is located at: %s' %
      haxby_dataset.func[0])

# Second, load the labels
import numpy as np

session_target = np.recfromcsv(haxby_dataset.session_target[0], delimiter=" ")
haxby_labels = session_target['labels']

import matplotlib.pyplot as plt
from nilearn.input_data import NiftiLabelsMasker




# Smooth the data
from nilearn import image
fmri_filename = haxby_dataset.func[0]
fmri_img = image.smooth_img(fmri_filename, fwhm=6)

# Plot the mean image
from nilearn.plotting import plot_epi
mean_img = image.mean_img(fmri_img)
plot_epi(mean_img, title='Smoothed mean EPI', cut_coords=(36, -27, 66))


from scipy import stats
fmri_data = fmri_img.get_data()

x_shape, y_shape, z_shape, t_shape = fri_data.reshape(x_shape * y_shape * z_shape, t_shape)

condition_mask = np.logical_or(labels['labels'] == b'face',
                               labels['labels'] == b'cat')
target = haxby_labels[condition_mask]

#######################
## Fit the stab_lasso #
#######################

from stab_lasso import StabilityLasso


stab_lasso = StabilityLasso(theta = 0.1,
                            n_split = 100,
                            ratio_split = 0.5,
                            n_clusters = 'auto',
                            model_selection='univariate',
                            random_state = 1)

# train = 
# stab_lasso.fit(               


