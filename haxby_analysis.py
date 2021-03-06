"""
Simple example of decoding: the Haxby data
==============================================

Here is a simple example of decoding, reproducing the Haxby 2001
study on a face vs cat discrimination task in a mask of the ventral
stream.
"""

###########################################################################
# Retrieve and load the Haxby dataset
import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.feature_extraction import image
from nilearn import datasets
from nilearn.input_data import NiftiMasker
from nilearn.image import mean_img
from nilearn.plotting import plot_stat_map, show
from nilearn import plotting
from sklearn.metrics import precision_recall_curve

from stab_lasso import StabilityLasso

haxby_dataset = datasets.fetch_haxby()
# Load the behavioral labels
# Load target information as string and give a numerical identifier to each
labels = np.recfromcsv(haxby_dataset.session_target[0], delimiter=" ")

# scikit-learn >= 0.14 supports text labels. You can replace this line by:
# target = labels['labels']
_, target = np.unique(labels['labels'], return_inverse=True)


# Keep only data corresponding to faces or cats
condition_mask = np.logical_or(labels['labels'] == b'face',
                               labels['labels'] == b'house')
target = target[condition_mask]
sessions = labels['chunks'][condition_mask]


###########################################################################
# Prepare the data: apply the mask
mask_filename = haxby_dataset.mask
# For decoding, standardizing is often very important
nifti_masker = NiftiMasker(mask_img=mask_filename, standardize=True,
                           memory='cache', verbose=10)

func_filename = haxby_dataset.func[0]

fmri_masked = nifti_masker.fit_transform(func_filename)

# Restrict the classification to the face vs cat discrimination
fmri_masked = fmri_masked[condition_mask]

# Compute connectivity matrix: which voxel is connected to which
mask = nifti_masker.mask_img_.get_data()
shape = mask.shape
connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1],
                                   n_z=shape[2], mask=mask)

###########################################################################
# Univariate testing on all the sessions

f_score, p_val = f_classif(fmri_masked, target)
p_val_img = nifti_masker.inverse_transform(p_val)
bonferroni_thr = .05 / mask.sum()

pseudo_truth = p_val < bonferroni_thr
pseudo_truth_img = nifti_masker.inverse_transform(pseudo_truth)

###########################################################################
# The decoding

# Use a reduced dataset
session_mask = np.in1d(sessions, [1, 3, 5, 7, 9, 11])
this_data = fmri_masked[session_mask]
this_target = target[session_mask]

model = StabilityLasso(theta=.1, n_split=10, n_clusters=2000)
model.fit(this_data, this_target, connectivity=connectivity)

###########################################################################
# Back to neuroimaging
"""
beta_img = nifti_masker.inverse_transform(model._soln)

beta_img.to_filename('soln.nii.gz')

display = plotting.plot_stat_map(beta_img,
                                 bg_img=haxby_dataset['anat'][0],
                                 cut_coords=(33, -34, -16))

# Hack: avoid outlining the brain by negating
display.add_contours(pseudo_truth_img,
                     levels=[.5], colors='c')

display.savefig('beta.png')
display.close()
"""
###########################################################################
# Compute prediction scores using cross-validation

from sklearn.cross_validation import KFold, LabelKFold

cv = LabelKFold(sessions, n_folds=6)
cv_scores = []

for train, test in cv:
    model.fit(fmri_masked[train], target[train], connectivity=connectivity)
    prediction = 3 + (model.predict(fmri_masked[test]) > 3.5)
    cv_scores.append(np.sum(prediction == target[test])
                     / float(np.size(target[test])))
print(cv_scores)
print(np.mean(cv_scores))
"""
###########################################################################
# Retrieve the discriminating weights and save them

# Retrieve the model discriminating weights
coef_ = model.coef_

# Reverse masking thanks to the Nifti Masker
coef_img = nifti_masker.inverse_transform(coef_)

# Save the coefficients as a Nifti image
coef_img.to_filename('haxby_model_weights.nii')

###########################################################################
# Visualize the discriminating weights over the mean EPI

mean_epi = mean_img(func_filename)
plot_stat_map(coef_img, mean_epi, title="SVM weights", display_mode="yx")
"""

###########################################################################
# Small sample recovery experiment

from sklearn.cross_validation import LabelShuffleSplit
from sklearn import metrics

# run a model on all the data
model.fit(fmri_masked, target, connectivity=connectivity)

for proportion in [1. / 6, 1./4, 1./3, 1./2]:
    slo = LabelShuffleSplit(sessions, n_iter=10, train_size=proportion,
                            random_state=0)
    # get the coefs:
    coef_all = model.coef_
    bin_coef_all = np.abs(coef_all) > np.percentile(np.abs(coef_all), 10)
    coefs = []
    for train, _ in slo:
        coefs.append(model.fit(fmri_masked[train], target[train],
                               connectivity=connectivity).coef_)

    auc = []
    for coef in coefs:
        fpr, tpr, _ = precision_recall_curve(bin_coef_all, np.abs(coef))
        auc.append(metrics.roc_auc_score(bin_coef_all, np.abs(coef)))

    print(proportion, np.mean(auc))

show()
