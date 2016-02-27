"""
=================================================
Example of pattern recognition on simulated data
=================================================

This example simulates data according to a very simple sketch of brain
imaging data and applies machine learning techniques to predict output
values.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, ndimage

from sklearn.utils import check_random_state

import nibabel


###############################################################################
# Function to generate data
def create_simulation_data(snr=0, n_samples=200, size=12, random_state=1,
                           modulation=False):
    generator = check_random_state(random_state)
    roi_size = 2
    smooth_X = 1
    ### Coefs
    w = np.zeros((size, size, size, 5))
    w[0:roi_size, 0:roi_size, 0:roi_size, 0] = -0.6
    w[-roi_size:, -roi_size:, 0:roi_size, 1] = 0.5
    w[0:roi_size, -roi_size:, -roi_size:, 2] = -0.6
    w[-roi_size:, 0:roi_size:, -roi_size:, 3] = 0.5
    w[(size - roi_size) // 2:(size + roi_size) // 2,
      (size - roi_size) // 2:(size + roi_size) // 2,
      (size - roi_size) // 2:(size + roi_size) // 2, 4] = 0.5
    if modulation:
        w = np.array([w_. ravel() for w_ in w.T])
    else:
        w = w.sum(-1).ravel()[np.newaxis]

    ### Generate smooth background noise
    X_ = generator.randn(n_samples, size, size, size)
    noise = []
    for i in range(n_samples):
        Xi = ndimage.filters.gaussian_filter(X_[i, :, :, :], smooth_X)
        Xi = Xi.ravel()
        noise.append(Xi)

    noise = np.array(noise)
    ### Generate the signal y and X
    y_ = generator.randn(n_samples, 1)
    if modulation:
        y = y_ * np.random.rand(n_samples, 5) ** 4
    else:
        y = y_

    X = np.dot(y, w)

    ## Generate the noise
    norm_noise = linalg.norm(X, 2) / np.exp(snr / 20.)
    noise_coef = norm_noise / linalg.norm(noise[:, w.sum(0) != 0], 2)
    noise *= noise_coef
 
    ### Mixing of signal + noise and splitting into train/test
    ## Old noising version
    X += noise
    X -= X.mean(axis=-1)[:, np.newaxis]
    X /= X.std(axis=-1)[:, np.newaxis]
    return X, np.ravel(y_), snr, noise, w.sum(0)[np.newaxis], size


def plot_slices(data, title=None):
    plt.figure(figsize=(5.5, 5.5))
    vmax = np.abs(data).max()
    n_slices = data.shape[2]
    for i in range(n_slices):
        plt.subplot(np.ceil(float(n_slices) / 3), 3, i + 1)
        plt.imshow(data[:, :, i], vmin=-vmax, vmax=vmax,
                  interpolation="nearest", cmap=plt.cm.RdBu_r)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(hspace=0.05, wspace=0.05, left=.03, right=.97, top=.9)
    if title is not None:
        plt.suptitle(title, y=.95)
