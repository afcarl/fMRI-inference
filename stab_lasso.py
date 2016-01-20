import numpy as np
from sklearn.linear_model import Lasso
from sklearn.cluster import FeatureAgglomeration, AgglomerativeClustering

import statsmodels.api as sm


def projection(X, k, connectivity):
    """
    Take the data, and returns a matrix, to reduce the dimension
    Returns, invP, (P.(X.T)).T and the underlying labels
    """
    n, p = X.shape
    ward = FeatureAgglomeration(
        linkage='ward', n_clusters=k, connectivity=connectivity)
    ward.fit(X)
    #
    labels = ward.labels_
    P = np.zeros((k, p))
    P[labels, np.arange(p)] = 1
    P_inv = P.copy().T
    s_array = P.sum(axis=0)
    P = P / s_array
    X_proj = P.dot(X.T).T
    # should be done through ward.transform, but there is an issue
    # with the normalization
    # X_proj = ward.transform(X)
    return P_inv, X_proj, labels


def pp_inv(clust):
    n_clusters = len(np.unique(clust))
    p = clust.size
    P = np.zeros((n_clusters, p))
    P[clust, np.arange(p)] = 1.
    P_inv = np.copy(P)
    P_inv = P_inv.T
    s_array = P.sum(axis=0)
    P = P / s_array
    return P, P_inv


def multivariate_split_pval(X, y, n_split, size_split, n_clusters,
                            beta_array, split_array, clust_array):
    """Main function to obtain p-values across splits """
    n, p = X.shape
    pvalues = np.ones((n_split, p))
    for i in range(n_split):
        # perform the split
        split = np.zeros(n, dtype='bool')
        split[split_array[i]] = True
        y_test = y[~split]
        X_test = X[~split]

        # projection
        P, P_inv = pp_inv(clust_array[i])

        # get the support 
        beta = beta_array[i]
        beta_proj = np.dot(P, beta)
        # this is very awkward
        model_proj = (beta_proj ** 2 > 0)
        model_proj_size = model_proj.sum()
        X_test_proj = np.dot(P, X_test.T).T
        X_model = X_test_proj[:, model_proj]

        # fit the model on test data to get p-values
        res = sm.OLS(y_test, X_model).fit()
        pvalues_proj = np.ones(n_clusters)
        pvalues_proj[model_proj] = np.clip(
            model_proj_size * res.pvalues, 0., 1.)

        pvalues[i] = np.dot(P_inv, pvalues_proj)

    if n_split > 1:
        pvalues_aggregated = pvalues_aggregation(pvalues)
    else:
        pvalues_aggregated = pvalues[0]
    return pvalues, pvalues_aggregated


def univariate_split_pval(X, y, n_split, size_split, n_clusters,
                         beta_array, split_array, clust_array):
    """Univariate p-values computation
    todo: replace permutations with analytical tests
    """
    n, p = X.shape
    pvalues = np.ones((n_split, p))
    n_perm = 10000
    for i in range(n_split):
        split = np.zeros(n, dtype='bool')
        split[split_array[i]] = True
        y_test = y[~split]
        X_test = X[~split]

        # projection
        P, P_inv = pp_inv(clust_array[i])

        X_test_proj = np.dot(P, X_test.T).T
        corr_perm = np.zeros((n_perm, n_clusters))
        for s in range(n_perm):
            perm = np.random.permutation(n - size_split)
            corr_perm[s] = np.dot(y_test.T, X_test_proj[perm])

        corr_perm = np.abs(corr_perm)
        corr_true = np.abs(np.dot(y_test.T, X_test_proj).reshape(
                (n_clusters)))

        pvalues_proj = 1. / n_perm * (corr_true < corr_perm).sum(axis=0)
        pvalues[i, :] = np.dot(P_inv, pvalues_proj)

    if n_split > 1:
        pvalues_aggregated = pvalues_aggregation(pvalues)
    else:
        pvalues_aggregated = pvalues[0]
    return pvalues, pvalues_aggregated


def pvalues_aggregation(pvalues, gamma_min=0.05):
    n_split, p = pvalues.shape
    kmin = max(1, int(gamma_min * n_split))
    pvalues_sorted = np.sort(pvalues, axis=0)[kmin:]
    gamma_array = 1. / n_split * (np.arange(kmin + 1, n_split + 1))
    pvalues_sorted = pvalues_sorted / gamma_array[:, np.newaxis]
    q = pvalues_sorted.min(axis=0)
    q *= 1 - np.log(gamma_min)
    q = q.clip(0., 1.)
    return q


class StabilityLasso(object):

    alpha = 0.05

    def __init__(self, y, X, theta, n_split=100, size_split=None,
                 n_clusters=None, connectivity=None):
        """

        Parameters
        ----------

        y : np.float(y)
            The target, in the model $y = X\beta$

        X : np.float((n, p))
            The data, in the model $y = X\beta$

        theta : np.float
            Coefficient factor of the L-1 penalty in
            $\text{minimize}_{\beta} \frac{1}{2} \|y_{splitted}-X_{clustered,split}\beta\|^2_2 + 
                \lambda\|\beta\|_1$
            where $\lambda = \theta \|X_{clustered, split}^Ty_{split}\|_\infty

        n_split : int
            Number of time we randomize the data

        size_split : int
            Size of the first part of the we are doing the selection on

        n_clusters : int
            Number of clusters in the clustering.
            Default is the number of voxels

        connectivity : np.array(p,p)
            Connectivity matrix of the data, used for the spatial clustering

        """
        self.y = y
        self.X = X
        n, p = X.shape
        self.theta = theta
        self.n_split = n_split

        if size_split is None:
            size_split = n
        self.size_split = size_split

        if n_clusters is None:
            n_clusters = p
        self._n_clusters = n_clusters
        self.connectivity = connectivity

    def fit(self, **lasso_args):
        X = self.X
        y = self.y
        n, p = X.shape
        n_split = self.n_split
        size_split = self.size_split
        n_clusters = self._n_clusters
        connectivity = self.connectivity
        theta = self.theta

        beta_array = np.zeros((n_split, p))
        split_array = np.zeros((n_split, size_split), dtype=int)
        clust_array = np.zeros((n_split, p), dtype=int)

        for i in range(n_split):
            split = np.random.choice(n, size_split, replace=False)
            split.sort()

            y_splitted, X_splitted = y[split], X[split]

            P_inv, X_proj, labels = projection(
                X_splitted, n_clusters, connectivity)

            alpha = theta * np.max(np.abs(np.dot(X_proj.T, y_splitted))) / n
            lasso_splitted = Lasso(alpha=alpha)
            lasso_splitted.fit(X_proj, y_splitted)

            beta_proj = lasso_splitted.coef_
            beta = np.dot(P_inv, beta_proj)

            beta_array[i] = beta
            split_array[i] = split
            clust_array[i] = labels

        beta = beta_array.mean(axis=0)
        self._soln = beta
        self._beta_array = beta_array
        self._split_array = split_array
        self._clust_array = clust_array

    def multivariate_split_pval(self):
        pvalues, pvalues_aggregated = multivariate_split_pval(
            self.X, self.y, self.n_split, self.size_split, self._n_clusters,
            self._beta_array, self._split_array, self._clust_array)
        self._pvalues = pvalues
        self._pvalues_aggregated = pvalues_aggregated
        return pvalues_aggregated

    def univariate_split_pval(self):
        pvalues, pvalues_aggregated = univariate_split_pval(
            self.X, self.y, self.n_split, self.size_split, self._n_clusters,
            self._beta_array, self._split_array, self._clust_array)
        self._pvalues = pvalues
        self._pvalues_aggregated = pvalues_aggregated
        return pvalues_aggregated

    def select_model_fwer(self, alpha):
        return np.where(self._pvalues_aggregated < alpha)[0]

    def select_model_fdr(self, q, normalize=True):
        pvalues = self._pvalues_aggregated
        p, = pvalues.shape
        pvalues_sorted = np.sort(pvalues) / np.arange(1, p + 1)
        newq = q / np.log(p)
        if (pvalues_sorted > newq).all():
            return []
        h = np.where(pvalues_sorted <= newq)[0][-1]
        bound = h * pvalues_sorted[h]
        return np.where(pvalues < bound)[0]
