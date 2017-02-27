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

ROI_SIZE = 1
SHAPE = (12, 12, 12)

###############################################################################
# Function to generate data
def create_simulation_data(snr=0, n_samples=200, shape=SHAPE, random_state=1,
                           modulation=False, roi_size=ROI_SIZE, smooth_X=1):
    generator = check_random_state(random_state)
    ### Coefs
    w = np.zeros(shape + (5,))
    w[0:roi_size, 0:roi_size, 0:roi_size, 0] = -0.6
    w[-roi_size:, -roi_size:, 0:roi_size, 1] = 0.5
    w[0:roi_size, -roi_size:, -roi_size:, 2] = -0.6
    w[-roi_size:, 0:roi_size:, -roi_size:, 3] = 0.5
    w[(shape[0] - roi_size) // 2:(shape[0] + roi_size) // 2,
      (shape[1] - roi_size) // 2:(shape[1] + roi_size) // 2,
      (shape[2] - roi_size) // 2:(shape[2] + roi_size) // 2, 4] = 0.5
    if modulation:
        w = np.array([w_. ravel() for w_ in w.T])
    else:
        w = w.sum(-1).ravel()[np.newaxis]

    ### Generate smooth background noise
    X_ = generator.randn(n_samples, shape[0], shape[1], shape[2])
    noise = []
    for i in range(n_samples):
        Xi = ndimage.filters.gaussian_filter(X_[i], smooth_X)
        Xi = Xi.ravel()
        noise.append(Xi)

    noise = np.array(noise)
    ### Generate the signal y and X
    y_ = generator.randn(n_samples, 1)
    if modulation:
        modulation_ = generator.rand(n_samples, 5)
        y = y_ * (.1 + .9 * (modulation_.T >= modulation_.max(1))).T
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
    return X, np.ravel(y_), snr, noise, w.sum(0)[np.newaxis], shape


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


def plot_row_slices(coefs):
    """ Given a the coefs dictionray, plot the slices in a row
    """
    plt.figure(figsize=(8, 8))
    # vmax = np.abs(coefs.values()).max()
    vmax = 10
    n_slices = coefs.values()[0].shape[-1]
    n_keys = len(coefs.keys())
    for q, key in enumerate(coefs.keys()):
        data = coefs[key]
        for i in range(n_slices):
            plt.subplot(n_keys, n_slices, i + n_slices * q + 1)
            plt.imshow(data[:, :, i], vmin=-vmax, vmax=vmax,
                       interpolation="nearest", cmap=plt.cm.RdBu_r)
            plt.xticks(())
            plt.yticks(())
        plt.text(x=-20, y=float(q) / n_keys - 1., s=key)
        print('%s, max: %f, min: %f' % (key, data.max(), data.min()))
    plt.subplots_adjust(hspace=0.01, wspace=0.1, left=.06, right=.99,
                        top=.99, bottom=.01)
