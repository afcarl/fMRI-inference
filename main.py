
import numpy as np
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

from plot_simulated_data import create_simulation_data, plot_slices
from stab_lasso import StabilityLasso
from sklearn.metrics import roc_curve


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
         rs=1,
         alpha=.05):
    size = 6

    k = int(size ** 3 / mean_size_clust)

    X, y, snr, noise, beta0, size = \
        create_simulation_data(snr, n_samples, size, rs)

    connectivity_ = connectivity(size)
    true_coeff = beta0 ** 2 > 0
    B = StabilityLasso(theta, n_split=n_split, ratio_split=split_ratio,
                       n_clusters=k, model_selection=model_selection)
    B.fit(X, y, connectivity_)
    beta = B._soln

    if model_selection == 'univariate':
        pvals = B.univariate_split_pval()
    elif model_selection == 'multivariate':
        pvals = B.multivariate_split_pval(X, y)
    else:
        raise ValueError("This model selection method doesn't exist")


    if model_selection == 'univariate':
        selected_model = B.select_model_fdr(alpha)
    elif model_selection == 'multivariate':
        # selected_model = B.select_model_fwer(alpha)
        selected_model = B.select_model_fdr(alpha)

    beta_corrected = np.zeros(size ** 3)
    if len(selected_model) > 0:
        beta_corrected[selected_model] = beta[selected_model]
        false_discovery = selected_model * (~true_coeff)
        true_discovery = selected_model * true_coeff
    else:
        false_discovery = np.array([])
        true_discovery  = np.array([])

    undiscovered = true_coeff.sum() - true_discovery.sum()

    fdr = (float(false_discovery.sum()) /
           max(1., float(selected_model.sum())))

    recall = float(true_discovery.sum()) / np.sum(true_coeff)

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
        coef_est = np.reshape(beta_corrected, [size, size, size])
        plot_slices(coef_est, title="Estimated")
        plot_slices(np.reshape(true_coeff, (size, size, size)),
                    title="Ground truth")
        plt.show()

    return fdr, recall, pvals, true_coeff


def multiple_test(n_test,
                  model_selection='multivariate',
                  n_samples=100,
                  n_split=30,
                  split_ratio=.4,
                  mean_size_clust=1,
                  theta=0.1,
                  snr=-10,
                  rs_start=1,
                  plot=False):
    fdr_array = []
    recall_array = []
    pvals = []
    true_coeffs = []
    
    for i in range(n_test):
        fdr, recall, pval, true_coeff = test(n_samples=n_samples,
                              n_split=n_split,
                              split_ratio=split_ratio,
                              mean_size_clust=mean_size_clust,
                              theta=theta,
                              snr=snr,
                              rs=rs_start + i,
                              print_results=False,
                              plot=plot)
        fdr_array.append(fdr)
        recall_array.append(recall)
        pvals.append(pval)
        true_coeffs.append(true_coeff)

    fpr, tpr, thresholds = roc_curve(
        np.concatenate(true_coeffs), 1 - np.concatenate(pvals))
    plt.figure()
    plt.plot(fpr, tpr,)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.show()

    return np.array(fdr_array), np.array(recall_array)

# r = np.random.randint(0, 200)

if __name__ == '__main__':
    fdr_array, recall_array = multiple_test(
        n_test=10, n_split=30, mean_size_clust=10, split_ratio=.5, plot=False)
    print('average fdr:', np.mean(fdr_array))
    print('average recall:', np.mean(recall_array))
    print('fwer:', np.mean(fdr_array > 0))
