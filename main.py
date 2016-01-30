
import numpy as np
from scipy.sparse import coo_matrix

from plot_simulated_data import *
from stab_lasso import *

import matplotlib.pyplot as plt

print "testmain"
# import pdb


def get_param(snr=100, n_samples=100, size=12, n_iterations=100):
    norm = []
    for i in range(n_iterations):
        if i % 10 == 0:
            print i
        X, y, snr, noise, beta0, size = \
            create_simulation_data(snr=-10, n_samples=100, size=15,
                                   random_state=i)
        v = np.dot(X.T, y)
        u = np.abs(v).max()
        norm.append(u)

    norm = sorted(norm)
    lam = norm[(9 * n_iterations) / 10]
    return lam


def connectivity(size):
    from sklearn.feature_extraction import image
    connectivity = image.grid_to_graph(n_x=size, n_y=size, n_z=size)
    return connectivity


def test(model_selection='multivariate',
         plot=False,
         plot_roc=False,
         print_results=True,
         n_samples=100,
         n_split=1,
         split_ratio=.4,
         mean_size_clust=1,
         theta=0.1,
         snr=-10,
         rs=0):
    size = 12
    p = size **3
    size_split = int(split_ratio * n_samples)
    k = int(p / mean_size_clust)

    X, y, snr, noise, beta0, size = \
        create_simulation_data(snr, n_samples, size, rs)
    co = connectivity(size)
    true_coeff = [i for i in range(p) if beta0[i] != 0]
    # lam = theta * np.max(np.abs(np.dot(X.T, y)))/k
    # print "Lambda : ", lam
    B = stab_lasso(y, X, theta, n_split=n_split, size_split=size_split,
                   n_clusters=k, connectivity=co)
    B.fit()
    
    beta_array = B._beta_array
    beta = B._soln

    if model_selection == 'univariate':
        pvals = B.univariate_split_pval()
    elif model_selection == 'multivariate':
        pvals = B.multivariate_split_pval()
    else:
        raise ValueError("This model selection method doesn't exist")

    true_model = np.where(beta0)[0]
    true_model_bool = np.zeros(p, dtype=bool)
    true_model_bool[true_model] = True

    if model_selection == 'univariate':
        selected_model = select_model_fdr(pvals, 0.1)
    elif model_selection == 'multivariate':
        selected_model = select_model_fdr(pvals, 0.1, normalize=False)

    beta_corrected = np.zeros(p)
    if len(selected_model) > 0:
        beta_corrected[selected_model] = beta[selected_model]
        false_discovery = selected_model[beta0[selected_model] == 0]
        true_discovery = selected_model[beta0[selected_model] != 0]
    else:
        false_discovery = np.array([])
        true_discovery  = np.array([])

    undiscovered = len(true_coeff) - true_discovery.shape[0]

    fdr = (float(false_discovery.shape[0]) /
           max(1., float(selected_model.shape[0])))

    if print_results:
        print("------------------- RESULTS -------------------")
        print("-----------------------------------------------")
        print("FDR : ", fdr)
        print("DISCOVERED FEATURES : "+ str(true_discovery.shape[0]))
        print("UNDISCOVERED FEATURES : "+ str(undiscovered))
        print("-----------------------------------------------")
        print("TRUE DISCOVERY")
        print("| Feature ID |       p-value      |")
        for i in true_discovery:
            print("|    "+ str(i).zfill(4)+ "    |   "+ str(pvals[i])+ "   |")
        print("-----------------------------------------------")
        print("FALSE DISCOVERY")
        print("| Feature ID |       p-value      |")
        for i in false_discovery:
            print("|    "+ str(i).zfill(4)+ "    |   "+ str(pvals[i])+ "   |")
        print("-----------------------------------------------")
    if plot:
        # Create masks for SearchLight. process_mask is the voxels where SearchLight
        # computation is performed. It is a subset of the brain mask, just to reduce
        # computation time.
        # mask = np.ones((size, size, size), np.bool)
        # mask_img = nibabel.Nifti1Image(mask.astype(np.int), np.eye(4))
        # process_mask = np.zeros((size, size, size), np.bool)
        # process_mask[:, :, 0] = True
        # process_mask[:, :, 5] = True
        # process_mask[:, :, 11] = True
        # process_mask_img = nibabel.Nifti1Image(process_mask.astype(np.int),
        #                                        np.eye(4))

        coefs = np.reshape(beta0, [size, size, size])
        coef_est = np.reshape(beta_corrected, [size, size, size])
        plot_slices(coef_est, title="Ground truth")
        plt.show()

    if plot_roc:
        
        if model_selection == 'univariate':
            model_bounds = np.sort(select_model_fdr_bounds(pvals))
        elif model_selection == 'multivariate':
            model_bounds = np.sort(select_model_fdr_bounds(pvals, normalize=False))

        plt_roc(model_bounds, true_model_bool)
        
    return fdr, pvals


def plt_roc(bounds, true_model):
    bounds_true = np.sort(bounds[true_model])
    bounds_false = np.sort(bounds[~true_model])
    size_true_model = np.sum(true_model)
    p, = np.shape(true_model)

    bounds_sorted = np.sort(bounds)
    roc_tdr = []
    roc_fdr = []
    for i in range(p):
            
        n_true = np.searchsorted(bounds_true, bounds_sorted[i], side='right')
        n_false = np.searchsorted(bounds_false, bounds_sorted[i], side='right')
        tdr = float(n_true) / size_true_model
        fdr = float(n_false) / (p-size_true_model)
        roc_tdr.append(tdr)
        roc_fdr.append(fdr)
        
    plt.plot(roc_fdr, roc_tdr, c='b')
    plt.plot(roc_fdr, roc_fdr, c='r')
    plt.show()


        
def multiple_test(n_test,
                  model_selection='multivariate',
                  n_samples=100,
                  n_split=1,
                  split_ratio=.4,
                  mean_size_clust=1,
                  theta=0.1,
                  snr=-10,
                  rs_start=0,
                  plot=False):
    fdr_array = []
    for i in range(n_test):
        if i % 10 == 0:
            print(i)

        fdr, _ = test(n_samples=n_samples,
                      n_split=n_split,
                      split_ratio=split_ratio,
                      mean_size_clust=mean_size_clust,
                      theta=theta,
                      snr=snr,
                      rs=rs_start + i,
                      print_results=False,
                      plot=plot)
        fdr_array.append(fdr)
    return fdr_array

# r = np.random.randint(0, 200)

# # Create data
# X, y, snr, _, coefs, size = \
#     create_simulation_data(snr=-10, n_samples=100, size=12, random_state = r)

# # Create masks for SearchLight. process_mask is the voxels where SearchLight
# # computation is performed. It is a subset of the brain mask, just to reduce
# # computation time.
# mask = np.ones((size, size, size), np.bool)
# mask_img = nibabel.Nifti1Image(mask.astype(np.int), np.eye(4))
# process_mask = np.zeros((size, size, size), np.bool)
# process_mask[:, :, 0] = True
# process_mask[:, :, 5] = True
# process_mask[:, :, 11] = True
# process_mask_img = nibabel.Nifti1Image(process_mask.astype(np.int), np.eye(4))


# coefs = np.reshape(coefs, [size, size, size])

# size_split = 80
# n_split = 20
# k = size**3 / 2
# co = connectivity(size)
# lam = 10.

# B = stab_lasso(y, X, lam, n_split=n_split, size_split=size_split, k=k, connectivity=co)
# B.fit()
# beta = B.soln

# coef_est = np.reshape(beta, [size, size, size])

# plot_slices(coef_est, title="Ground truth")
# plt.show()
