
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
    size = 40

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
        pvals = B.univariate_split_pval(X, y)
    elif model_selection == 'multivariate':
        pvals = B.multivariate_split_pval(X, y)
    else:
        raise ValueError("This model selection method doesn't exist")


    if model_selection == 'univariate':
        selected_model = B.select_model_fdr(alpha)
    elif model_selection == 'multivariate':
        # selected_model = B.select_model_fwer(alpha)
        selected_model = B.select_model_fdr(alpha, normalize = False)

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
        print("DISCOVERED FEATURES : ", true_discovery.sum())
        print("UNDISCOVERED FEATURES : ", undiscovered)
        print("-----------------------------------------------")
        print("TRUE DISCOVERY")
        print("| Feature ID |       p-value      |")
        for i in range(size ** 3):
            if true_discovery[i]:
                print("|   "+str(i).zfill(4)+"   |  "+str(pvals[i])+"  |")
        print("-----------------------------------------------")
        print("FALSE DISCOVERY")
        print("| Feature ID |       p-value      |")
        for i in range(size ** 3):
            if false_discovery[i]:
                print("|   "+str(i).zfill(4)+"   |  "+str(pvals[i])+"  |")
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
                  plot=False, 
                  alpha=.05):
    fdr_array = []
    recall_array = []
    pvals = []
    true_coeffs = []

    for i in range(n_test):
        fdr, recall, pval, true_coeff = test(
            model_selection=model_selection,
            n_samples=n_samples,
            n_split=n_split,
            split_ratio=split_ratio,
            mean_size_clust=mean_size_clust,
            theta=theta,
            snr=snr,
            rs=rs_start + i,
            print_results=False,
            plot=plot,
            alpha=alpha)
        fdr_array.append(fdr)
        recall_array.append(recall)
        pvals.append(pval)
        true_coeffs.append(true_coeff)

    return np.array(fdr_array), np.array(recall_array)


def experiment_nominal_control():
    for n_split in [1, 10]:
        for mean_size_clust in [1, 10]:
            fdr_array, recall_array = multiple_test(
                model_selection='univariate',
                n_test=20, n_split=n_split, mean_size_clust=mean_size_clust,
                split_ratio=.4, plot=False, alpha=.1)
            print('cluster_size %d, n_split %d' % (mean_size_clust, n_split))
            print('average fdr:', np.mean(fdr_array))
            print('average recall:', np.mean(recall_array))
            print('fwer:', np.mean(fdr_array > 0))


def experiment_roc_curve():
    # set various parameters
    n_samples = 100
    model_selection = 'multivariate'
    n_test = 20
    split_ratio = .4
    theta = 0.1
    snr = - 10
    rs_start = 1

    ax = plt.subplot(111)
    for n_split in [1, 10]:
        for mean_size_clust in [1, 10]:
            # collect results
            pvals = []
            true_coeffs = []
            for i in range(n_test):
                fdr, recall, pval, true_coeff = test(
                    model_selection=model_selection,
                    n_samples=n_samples,
                    n_split=n_split,
                    split_ratio=split_ratio,
                    mean_size_clust=mean_size_clust,
                    theta=theta,
                    snr=snr,
                    rs=rs_start + i,
                    print_results=False,
                    plot=False)
                pvals.append(pval)
                true_coeffs.append(true_coeff)
                n_clusters = pval.size / mean_size_clust

            fpr, tpr, thresholds = roc_curve(
                np.concatenate(true_coeffs), 1 - np.concatenate(pvals))
            ax.plot(fpr, tpr, label='n_split=%d, %d clusters' % (
                    n_split, n_clusters))
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc=4)
    ax.set_title('ROC curves for basic settings')
    plt.savefig('roc_curves.png')


def experiment_univariate_multivariate():
    pass


if __name__ == '__main__':
    experiment_nominal_control()
    # experiment_roc_curve()
    plt.show()
