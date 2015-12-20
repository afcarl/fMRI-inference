import numpy as np
from scipy.sparse import coo_matrix

from plot_simulated_data import *
from stab_lasso import *



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

def test(plot = False, n_samples= 100, n_split=1, split_ratio=1., mean_size_clust = 1, theta = 0.1, snr = -10, rs=0):

    size = 12
    size_split = int(split_ratio * n_samples)
    k = int(size ** 3 / mean_size_clust)
    
    X, y, snr, noise, beta0, size = \
            create_simulation_data(snr, n_samples, size, rs)
    co = connectivity(size)
    true_coeff = [i for i in range(size**3) if beta0[i] != 0]
    
    # lam = theta * np.max(np.abs(np.dot(X.T, y)))/k
    # print "Lambda : ", lam
    B = stab_lasso(y, X, theta, n_split=n_split, size_split=size_split, k=k, connectivity=co)
    B.fit()
    # print ("Model fitted")
    #I = B.intervals
    # P = B.active_pvalues
    beta_array = B._beta_array
    beta = B.soln
    pvals = B.split_pval()

    true_model = np.arange(size ** 3)[beta0 != 0]
    #selected_model = np.arange(size**3)[pvals != 1.]
    selected_model = B.select_model_fdr(0.1)

    beta_corrected = np.zeros(size**3)
    beta_corrected[selected_model] = beta[selected_model]

    falsediscovery = np.arange(size**3)[np.logical_and(pvals != 1.,beta0 == 0)]
    truediscovery = np.arange(size**3)[np.logical_and(pvals != 1., beta0 != 0)]
    undiscovered = np.arange(size**3)[np.logical_and(pvals == 1., beta0 != 0)]
    fdr = float(falsediscovery.shape[0]) / max(1., float(selected_model.shape[0]))


    print "------------------- RESULTS -------------------"
    print "-----------------------------------------------"
    print "FDR : ", fdr
    print "DISCOVERED FEATURES : ", truediscovery.shape[0]
    print "UNDISCOVERED FEATURES : ", undiscovered.shape[0]
    print "-----------------------------------------------"
    print "TRUE DISCOVERY"
    for i in truediscovery:
        print i, pvals[i]
    print "FALSE DISCOVERY"
    for i in falsediscovery:
        print i, pvals[i]
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

    return B, beta_array, beta, pvals


def multiple_test(n_test):
    beta_array = []
    for i in range(n_test):
        if i % 10 == 0:
            print i
        
        beta_array.append(test(n_samples=150, n_split=50, split_ratio=0.8, rs=i))
        
    return beta_array
    
    

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
