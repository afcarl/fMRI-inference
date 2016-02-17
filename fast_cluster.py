""" Utilities for fast clustering

Sandbox for a future library

Author: Bertrand Thirion, Gael Varoquaux, Andres Hoyos idrobo, 2014
"""

import warnings
import numpy as np
from joblib import Parallel, delayed, Memory
from scipy.sparse import csgraph, coo_matrix, dia_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.base import BaseEstimator
from sklearn import clone
from nilearn.input_data import NiftiMasker
from sklearn.utils import check_array
from base_clustering import ClusteringTransformer

import matplotlib.pyplot as plt

#@profile
def _compute_weights(nifti_masker, data):
    """Compute the weights corresponding to all edges in the mask"""
    fmri = nifti_masker.inverse_transform(data).get_data()
    weight_deep = np.sum(np.diff(fmri, axis=2) ** 2, axis=-1)
    weight_right = np.sum(np.diff(fmri, axis=1) ** 2, axis=-1)
    weight_down = np.sum(np.diff(fmri, axis=0) ** 2, axis=-1)

    weight = np.hstack((weight_deep.ravel(),
                        weight_right.ravel(),
                        weight_down.ravel()))

    return weight


def _create_ordered_edges(nifti_masker, data):
    """ Create the edges set, the correspoding weight,
    and order them by increasing values"""
    # Compute connectivity matrix: which voxel is connected to which
    mask = nifti_masker.mask_img_.get_data()
    n_x, n_y, n_z = mask.shape
    n_voxels = n_x * n_y * n_z

    # The indices of the edges
    vertices = np.arange(n_voxels).reshape((n_x, n_y, n_z))

    edges_deep = np.vstack((vertices[:, :, :-1].ravel(),
                            vertices[:, :, 1:].ravel()))
    edges_right = np.vstack((vertices[:, :-1].ravel(),
                             vertices[:, 1:].ravel()))
    edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))
    edges = np.hstack((edges_deep, edges_right, edges_down))

    # Masking
    edges_deep_mask = np.logical_and(mask[:, :, :-1].ravel(),
                                     mask[:, :, 1:].ravel())
    edges_right_mask = np.logical_and(mask[:, :-1].ravel(),
                                      mask[:, 1:].ravel())
    edges_down_mask = np.logical_and(mask[:-1].ravel(), mask[1:].ravel())
    edges_mask = np.hstack((edges_deep_mask, edges_right_mask,
                            edges_down_mask))

    # Image-level operation to compute weights on the edges
    weight = _compute_weights(nifti_masker, data)

    # Apply the mask
    weight = weight[edges_mask]
    edges = edges[:, edges_mask]

    # Reorder the indices of the graph
    max_index = edges.max()
    order = np.searchsorted(np.unique(edges.ravel()), np.arange(max_index + 1))
    edges = order[edges]

    return edges, weight, edges_mask



def _fast_nn_connectivity(connectivity, return_weight=False, thr=1.):
    """ Reduce the connectivity to nearest neighbor connectivity"""
    n_voxels = connectivity.shape[0]
    connectivity_ = coo_matrix(
        (1. / connectivity.data, connectivity.nonzero()),
        (n_voxels, n_voxels)).tocsr()
    inv_max = dia_matrix((1. / connectivity_.max(axis=0).toarray()[0], 0),
                         shape=(n_voxels, n_voxels))
    connectivity_ = inv_max * connectivity_
    edge_mask = connectivity_.data > thr - 1.e-7
    j_idx = connectivity_.nonzero()[1][edge_mask]
    i_idx = connectivity_.nonzero()[0][edge_mask]

    if return_weight:
        weight = connectivity.data[edge_mask]
    else:
        weight = np.ones_like(j_idx)

    edges = np.array((i_idx, j_idx))
    nn_connectivity = coo_matrix(
        (weight, edges), (n_voxels, n_voxels))

    return nn_connectivity


def _nn_cluster_and_reduce(connectivity, data, n_clusters=None, random=False):
    """ Cluster according to nn and reduce the data and connectivity"""
    if n_clusters == None:
        n_clusters = 1
    nn_connectivity = _fast_nn_connectivity(connectivity)
    n_voxels = connectivity.shape[0]
    n_labels = n_voxels - (nn_connectivity + nn_connectivity.T).nnz / 2

    if n_labels < n_clusters:
        # cut some links to achieve the desired number of clusters
        alpha = n_voxels - n_clusters
        nn_connectivity = nn_connectivity + nn_connectivity.T
        edges_ = np.array(nn_connectivity.nonzero())
        plop = edges_[0] - edges_[1]
        select = np.argsort(plop)[:alpha]

        nn_connectivity = coo_matrix(
            (np.ones(2 * alpha),
             np.hstack((edges_[:, select], edges_[::-1, select]))),
            (n_voxels, n_voxels))

    # clustering
    n_labels, labels = csgraph.connected_components(nn_connectivity)
    incidence = _random_incidence(labels, n_labels, random)

    # reduced data
    r_data = (incidence * data.T).T
    r_connectivity = (incidence * connectivity) * incidence.T
    r_connectivity = r_connectivity - dia_matrix(
        (r_connectivity.diagonal(), 0), shape=(r_connectivity.shape))
    i_idx, j_idx = r_connectivity.nonzero()
    data_ = np.maximum(1.e-6, np.sum(
        (r_data[:, i_idx] - r_data[:, j_idx]) ** 2, 0))
    r_connectivity.data = data_
    assert (r_connectivity.data > 0).all()
    return r_connectivity, r_data, labels


def _random_incidence(labels, n_labels, random=False):
    """ Given a labelling, geenrate a randomized incidence matrix """
    n_voxels = len(labels)
    if random:
        weight = np.random.rand(n_voxels)
    else:
        weight = np.ones(n_voxels)
    incidence = coo_matrix(
        (weight, (labels, np.arange(n_voxels))),
        shape=(n_labels, n_voxels), dtype=np.float32).tocsc()
    inv_sum_col = dia_matrix(
        (np.array(1. / incidence.sum(1)).squeeze(), 0),
        shape=(n_labels, n_labels))

    return inv_sum_col * incidence


def recursive_nn(connectivity, data, n_clusters=None, n_iter=10, random=False):
    labels = np.arange(connectivity.shape[0])

    if n_clusters == None:
        n_clusters = 1
    n_labels = connectivity.shape[0]

    for i in range(n_iter):
        connectivity, data, r_labels = _nn_cluster_and_reduce(
            connectivity, data, n_clusters, random)
        labels = r_labels[labels]
        n_labels = connectivity.shape[0]

        if n_labels <= n_clusters:
            break

    return connectivity.shape[0], labels


def fast_cluster_nopercol(nifti_masker, data, n_clusters=500, random=False):
    """Attempts to implement a method that avoids percolation"""
    n_voxels = nifti_masker.mask_img_.get_data().sum()

    edges, weight, edges_mask = _create_ordered_edges(
        nifti_masker, data)

    connectivity = coo_matrix(
        (weight, edges), (n_voxels, n_voxels)).tocsr()
    connectivity = (connectivity + connectivity.T)

    n_labels, labels = recursive_nn(connectivity, data, n_clusters=n_clusters,
                                    random=random)
    return n_labels, labels

#@profile
def single_linkage(nifti_masker, data, n_clusters):
    """Single linkage clustering"""
    n_voxels = nifti_masker.mask_img_.get_data().sum()
    edges, weight, edges_mask = _create_ordered_edges(
        nifti_masker, data)
    connectivity = coo_matrix(
        (weight, edges), (n_voxels, n_voxels))
    connectivity = connectivity + connectivity.T

    mst = minimum_spanning_tree(connectivity)
    weight = mst.data
    mst.data = np.ones_like(mst.data)
    edges = np.array(mst.nonzero())

    index = np.argsort(weight)
    edges = edges[:, index[:-n_clusters]]
    weight = np.ones_like(index[:-n_clusters])

    mst = coo_matrix((weight, edges), (n_voxels, n_voxels))

    n_labels, labels = csgraph.connected_components(mst)
    # print n_clusters, n_labels

    return n_labels, labels


def random_single_linkage(nifti_masker, data, n_clusters):
    """Single linkage clustering with random selection"""
    n_voxels = nifti_masker.mask_img_.get_data().sum()
    edges, weight, edges_mask = _create_ordered_edges(
        nifti_masker, data)
    connectivity = coo_matrix(
        (weight, edges), (n_voxels, n_voxels))
    # connectivity = connectivity + connectivity.T

    mst = minimum_spanning_tree(connectivity)
    mst.data = np.ones_like(mst.data)
    edges, weight = np.array(mst.nonzero()), mst.data

    # do not select terminal nodes
    n_iter = 4
    for i in range(n_iter):
        singletons = np.asarray((mst + mst.T).sum(axis=0) == 1).ravel()

        edges_mask = singletons[edges].max(axis=0) > 0
        select = np.hstack((
                np.ones(len(weight) - edges_mask.sum() - n_clusters / n_iter),
                np.zeros(n_clusters / n_iter))).astype(np.bool)
        np.random.shuffle(select)
        edges_mask[edges_mask == 0] = select
        edges, weight = edges[:, edges_mask], np.ones(edges_mask.sum())
        mst = coo_matrix((weight, edges), (n_voxels, n_voxels))

    n_labels, labels = csgraph.connected_components(mst)

    return n_labels, labels





class ReNN(BaseEstimator, ClusteringTransformer):
    """
    """

    def __init__(self, linkage='fast', n_clusters=5000, masker=None,
                 standardize=True, smoothing_fwhm=None, target_affine=None,
                 target_shape=None, mask_strategy='epi', memory=None,
                 memory_level=0, verbose=0, n_jobs=1, random=False,
                 scaling=False):

        self.scaling = scaling
        self.linkage = linkage
        self.n_clusters = n_clusters
        self.masker = masker
        self.mask_strategy = mask_strategy
        self.standardize = standardize
        self.smoothing_fwhm = smoothing_fwhm
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random = random

    def fit(self, X, y=None):
        """
        """
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'],
                        ensure_min_features=2, estimator=self)

        self.memory_ = Memory(cachedir=self.memory, verbose=self.verbose)

        if self.linkage == 'fast':
            n_labels, labels = fast_cluster_nopercol(
                self.masker, X, n_clusters=self.n_clusters, random=self.random)

        elif self.linkage == 'random_single':
            n_labels, labels = random_single_linkage(self.masker, X,
                                                     n_clusters=self.n_clusters)

        elif self.linkage == 'single':
            n_labels, labels = single_linkage(self.masker, X,
                                              n_clusters=self.n_clusters)
        self.n_labels_ = n_labels
        self.labels_ = labels
        self._check_labels_and_sizes()

        return self




### XXX Deprecated functions ###################################################
# Used only for visualization of the method
from base_clustering import _check_parcelation_results

def fmri_reduction(data, labels, return_mat=False):
    """Fast cluster-based reduction of data array """
    n_voxels = data.shape[-1]
    n_parcels = len(np.unique(labels))

    parcellation_masks = coo_matrix(
        (np.ones(n_voxels), (labels, np.arange(n_voxels))),
        shape=(n_parcels, n_voxels),
        dtype=np.float32).tocsc()

    inv_sum_col = dia_matrix(
        (np.array(1. / parcellation_masks.sum(1)).squeeze(), 0),
        shape=(n_parcels, n_parcels))

    parcellation_masks = inv_sum_col * parcellation_masks
    fmri_reduced = parcellation_masks * data.T
    if return_mat:
        return fmri_reduced.T, parcellation_masks
    return fmri_reduced.T


def fmri_compression(data, labels, n_clusters):
    """Fast cluster-based compression of data array"""
    labels = _check_parcelation_results(labels, n_clusters)
    fmri_reduced = fmri_reduction(data, labels)
    fmri_compressed = np.array(fmri_reduced.T[labels])
    return fmri_compressed


import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map, plot_roi, plot_epi
from nilearn import datasets
from nilearn.input_data import NiftiMasker

if __name__ == '__main__':

    plt.close('all')

    data_dir = '/volatile/andres/brain_codes/DATA/nilearn_data'
    data_dir = None
    dataset = datasets.fetch_haxby(data_dir=data_dir, n_subjects=1)

    masker = NiftiMasker(mask_strategy='epi', smoothing_fwhm=6, memory='cache')
    X = masker.fit_transform(dataset.func[0])

    # cluster = KMeans(n_clusters=5)
    cluster = ReNN(masker=masker, scaling=False, n_clusters=2000,
                   linkage='fast',
                             # linkage='single',
                             # random=True,
                             )

    # # Test random projection
    # random_projection = SRandomProjections(n_clusters=100)
    # Xred = random_projection.fit_transform(X)

    Xred = cluster.fit_transform(X[0: 10])
    Xcomp = cluster.inverse_transform(Xred)

    cut_coords = (10, -10, 0)

    plot_epi(masker.inverse_transform(Xcomp[0]), title='compressed',
             display_mode='ortho', cut_coords=cut_coords)
    plot_epi(masker.inverse_transform(X[0]), title='original',
             display_mode='ortho', cut_coords=cut_coords)

     # Shuffle the labels (for better visualization):
    labels = cluster.labels_
    permutation = np.random.permutation(labels.shape[0])
    labels = permutation[labels]
    labels_img_ = masker.inverse_transform(labels)

    plot_stat_map(labels_img_, bg_img=dataset.anat[0], title='clusters',
                  display_mode='ortho', cut_coords=cut_coords, colorbar=False)
    plt.show()

    np.testing.assert_almost_equal(Xred[0], cluster.transform(Xcomp)[0])


