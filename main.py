
import numpy as np
from scipy.sparse import coo_matrix

from plot_simulated_data import *
from stab_lasso import *


print "testmain"
# import pdb


def get_param(snr = 100, n_samples = 100, size = 12, n_iterations=100):
    norm = []
    for i in range(n_iterations):
        
        if i % 10 == 0:
            print i
        X, y, snr, noise, beta0, size = \
            create_simulation_data(snr=-10, n_samples=100, size=15, random_state = i)
        v = np.dot(X.T, y)
        u = np.abs(v).max()
        norm.append(u)

    norm = sorted(norm)
    lam = norm[(9*n_iterations)/10]
    return lam
    

def connectivity(size):
    indices = np.arange(size**3).reshape((size, size, size))
    connectivity = []
    directions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], \
                           [-1, 0, 0], [0, -1, 0], [0, 0, -1]])

    
    for u in range(size):
        for v in range(size):
            for w in range(size):
                neighbors = directions + np.array([u, v, w])
                neighbors = [t for t in neighbors
                             if (np.all(t < size) and np.all(t >= 0))]
                id_neighbors = [indices[tuple(n)] for n in neighbors]
                connectivity.append(id_neighbors)

    id_i = [c for c in range(len(connectivity)) for f in range(len(connectivity[c])) ]
    id_j = [f for c in connectivity for f in c ]
    data_sparse = np.ones(len(id_i))
    connectivity = coo_matrix((data_sparse, (id_i, id_j)), (size**3, size**3))
    return connectivity

def test(model_selection='multivariate',
         plot = False,
         print_results = True,
         n_samples= 100,
         n_split=1,
         split_ratio=.4,
         mean_size_clust = 1,
         theta = 0.1,
         snr = -10,
         rs=0):

    size = 12
    size_split = int(split_ratio * n_samples)
    k = int(size ** 3 / mean_size_clust)
    
    X, y, snr, noise, beta0, size = \
            create_simulation_data(snr, n_samples, size, rs)
    co = connectivity(size)
    true_coeff = [i for i in range(size**3) if beta0[i] != 0]
    
    # lam = theta * np.max(np.abs(np.dot(X.T, y)))/k
    # print "Lambda : ", lam
    B = stab_lasso(y, X, theta, n_split=n_split, size_split=size_split, n_clusters=k, connectivity=co)
    B.fit()
    # print ("Model fitted")
    #I = B.intervals
    # P = B.active_pvalues
    beta_array = B._beta_array
    beta = B._soln

    if model_selection == 'univariate':
        pvals = B.univariate_split_pval()
    elif model_selection == 'multivariate':
        pvals = B.multivariate_split_pval()
    else:
        raise ValueError("This model selection method doesn't exist")

    true_model = np.arange(size ** 3)[beta0 != 0]
    #selected_model = np.arange(size**3)[pvals != 1.]

    if model_selection == 'univariate':
        selected_model = B.select_model_fdr(0.1)
    elif model_selection == 'multivariate':
        selected_model = B.select_model_fdr(0.1, normalize=False)

    beta_corrected = np.zeros(size**3)
    if len(selected_model) > 0:
        beta_corrected[selected_model] = beta[selected_model]
        falsediscovery = selected_model[beta0[selected_model] == 0]
        truediscovery = selected_model[beta0[selected_model] != 0]
    else:
        falsediscovery = np.array([])
        truediscovery  = np.array([])

    undiscovered = len(true_coeff) - truediscovery.shape[0]
    fdr = float(falsediscovery.shape[0]) / max(1., float(selected_model.shape[0]))

    
    if print_results:
        print "------------------- RESULTS -------------------"
        print "-----------------------------------------------"
        print "FDR : ", fdr
        print "DISCOVERED FEATURES : ", truediscovery.shape[0]
        print "UNDISCOVERED FEATURES : ", undiscovered
        print "-----------------------------------------------"
        print "TRUE DISCOVERY"
        print "| Feature ID |       p-value      |"
        for i in truediscovery:
            print "|   ", str(i).zfill(4), "   |  ", pvals[i], "  |"
        print "-----------------------------------------------"
        print "FALSE DISCOVERY"
        print "| Feature ID |       p-value      |"
        for i in falsediscovery:
            print "|   ", str(i).zfill(4), "   |  ", pvals[i], "  |"
        print "-----------------------------------------------"
    
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
        # process_mask_img = nibabel.Nifti1Image(process_mask.astype(np.int), np.eye(4))


        coefs = np.reshape(beta0, [size, size, size])
        coef_est = np.reshape(beta_corrected, [size, size, size])

        plot_slices(coef_est, title="Ground truth")
        plt.show()

    return fdr, pvals


def multiple_test(n_test,
                  model_selection='multivariate',
                  n_samples= 100,
                  n_split=1,
                  split_ratio=.4,
                  mean_size_clust = 1,
                  theta = 0.1,
                  snr = -10,
                  rs_start = 0):
    fdr_array = []
    for i in range(n_test):
        if i % 10 == 0:
            print i
        fdr, _ = test(n_samples=n_samples,
                      n_split=n_split,
                      split_ratio=split_ratio,
                      mean_size_clust=mean_size_clust,
                      theta=theta,
                      snr=snr,
                      rs = rs_start + i,
                      print_results = False)
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
