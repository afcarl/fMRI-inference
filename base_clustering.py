""" Unifying the interface for the clustering features agglomeration.
The idea is to include scaling (sqrt of the size of the cluster), all checking
functions and other minor functions.

Author: Andres Hoyos Idrobo
"""

import numpy as np
from sklearn import clone
from sklearn.random_projection import BaseRandomProjection
from sklearn.feature_extraction import image
import time
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted
from scipy import ndimage
from sklearn.utils import check_array
from sklearn.cluster import AgglomerativeClustering
from joblib import Memory
from sklearn.feature_extraction import image



def _scaling(X, sizes):
    return X * np.sqrt(sizes)

def _inv_scaling(X, sizes):
    return X / np.sqrt(sizes)


def _check_parcelation_results(labels, n_clusters):
    """ This function
    """
    labels_name, inverse = np.unique(labels, return_inverse=True)
    if n_clusters != labels_name.shape[0]:
        new_labels = np.arange(labels_name.shape[0])
        labels_ = new_labels[inverse]
    else:
        labels_ = labels
    return labels_



class ClusteringTransformer(TransformerMixin):
    """ Class for all clustering methods in this experiment
    """

    def _check_labels_and_sizes(self):
        """Checking and returning the labels and sizes
        """

        check_is_fitted(self, "labels_")

        self.labels_ = _check_parcelation_results(self.labels_,
                                                  self.n_clusters)
        sizes = np.bincount(self.labels_)
        sizes = sizes[sizes > 0]

        self.n_clusters_ = np.unique(self.labels_).shape[0]
        self.sizes_ = sizes
        return self


    def transform(self, X, y=None, pooling_func=np.mean):
        """
        We perform the for in the sample direction, because in our case the
        n_clusters > n_samples
        """

        check_is_fitted(self, 'labels_')
        unique_labels = np.unique(self.labels_)

        # Xred = np.empty((X.shape[0], unique_labels.size))
        # for i, this_X in enumerate(X):
        #     Xred[i, :] = ndimage.measurements.mean(this_X, labels=self.labels_,
        #                                            index=unique_labels)

        nX = []
        for l in unique_labels:
            nX.append(pooling_func(X[:, self.labels_ == l], axis=1))
        Xred =  np.array(nX).T

        if self.scaling:
            Xred = _scaling(Xred, self.sizes_)
        return Xred

    def inverse_transform(self, Xred):
        """
        """
        if self.labels_ is None:
            warnings.warn('Please use fit first')
        labels = _check_parcelation_results(self.labels_, self.n_clusters)

        unil, inverse = np.unique(self.labels_, return_inverse=True)
        if self.scaling:
            Xred = _inv_scaling(Xred, self.sizes_)
        return  Xred[..., inverse]



################################################################################
# My feature agglomeration
################################################################################
class MyFeatureAgglomeration(AgglomerativeClustering, ClusteringTransformer):
    """
    """
    def __init__(self, n_clusters=2, affinity="euclidean",
                 memory=Memory(cachedir=None, verbose=0),
                 connectivity=None, compute_full_tree='auto', linkage='ward',
                 pooling_func=np.mean, masker=None, scaling=False):

        super(MyFeatureAgglomeration, self).__init__(
            n_clusters=n_clusters, memory = memory, connectivity=connectivity,
            compute_full_tree=compute_full_tree, linkage=linkage,
            affinity=affinity, pooling_func=pooling_func)

        self.masker = masker
        self.scaling = scaling


    def fit(self, X, y=None, **fit_params):
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'],
                        ensure_min_features=2, estimator=self)

        if self.connectivity is None:
            mask_data = self.masker.mask_img_.get_data().astype(np.bool)
            n_x, n_y, n_z = mask_data.shape
            self.connectivity = image.grid_to_graph(n_x=n_x, n_y=n_y, n_z=n_z,
                                                    mask=mask_data)
        # print X.shape, self.connectivity.shape
        AgglomerativeClustering.fit(self, X.T, **fit_params)

        self._check_labels_and_sizes()

        return self


################################################################################
# Clustering utils
################################################################################
def _setparams_clustering(method, masker, n_clusters, crop=False):
    """Setting the parameters of the clustering method

    method : sklearn clustering-like or Random Projections
    masker : NiftiMasker
    n_clusters : int
    crop: bool, only for slic

    """
    method = clone(method)
    if hasattr(method, 'n_clusters'):
        method.set_params(**{'masker': masker, 'n_clusters': n_clusters})
        if hasattr(method, 'crop'):
            method.set_params(crop=crop)
    else:
        method.set_params(n_components=n_clusters)

    return method


def _fit_method(X, method, n_clusters, masker, crop=False):
    """Fit the clustering method

    method : sklearn clustering-like or RandomProjections
    n_clusters : Int
    masker : NiftiMasker
    """

    n_voxels = X.shape[1]
    method =  _setparams_clustering(method, masker, n_clusters, crop)

    ti = time.time()
    method.fit(X)
    to = time.time() - ti

    return method, to
