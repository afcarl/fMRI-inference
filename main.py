
import numpy as np
from scipy.sparse import coo_matrix

from plot_simulated_data import *
from stab_lasso import StabilityLasso


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
         print_results=True,
         n_samples=100,
         n_split=1,
         split_ratio=.4,
         mean_size_clust=1,
         theta=0.1,
         snr=-10,
         rs=0):
    size = 12
    size_split = int(split_ratio * n_samples)
    k = int(size ** 3 / mean_size_clust)

    X, y, snr, noise, beta0, size = \
        create_simulation_data(snr, n_samples, size, rs)
    co = connectivity(size)
    true_coeff = [i for i in range(size ** 3) if beta0[i] != 0]
    B = StabilityLasso(y, X, theta, n_split=n_split, size_split=size_split,
                       n_clusters=k, connectivity=co)
    B.fit()
    beta = B._soln

    if model_selection == 'univariate':
        pvals = B.univariate_split_pval()
    elif model_selection == 'multivariate':
        pvals = B.multivariate_split_pval()
    else:
        raise ValueError("This model selection method doesn't exist")

    if model_selection == 'univariate':
        selected_model = B.select_model_fdr(0.1)
    elif model_selection == 'multivariate':
        selected_model = B.select_model_fdr(0.1, normalize=False)

    beta_corrected = np.zeros(size ** 3)
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
        print("DISCOVERED FEATURES : ", true_discovery.shape[0])
        print("UNDISCOVERED FEATURES : ", undiscovered)
        print("-----------------------------------------------")
        print("TRUE DISCOVERY")
        print("| Feature ID |       p-value      |")
        for i in true_discovery:
            print("|   ", str(i).zfill(4), "   |  ", pvals[i], "  |")
        print("-----------------------------------------------")
        print("FALSE DISCOVERY")
        print("| Feature ID |       p-value      |")
        for i in false_discovery:
            print("|   ", str(i).zfill(4), "   |  ", pvals[i], "  |")
        print("-----------------------------------------------")
    if plot:
        coefs = np.reshape(beta0, [size, size, size])
        coef_est = np.reshape(beta_corrected, [size, size, size])
        plot_slices(coef_est, title="Ground truth")
        plt.show()

    return fdr, pvals


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


if __name__ == '__main__':
    print(multiple_test(10))
