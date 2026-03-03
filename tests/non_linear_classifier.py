import matplotlib.pyplot as plt
import numpy as np
from decodanda import Decodanda, generate_synthetic_data
from decodanda.utilities import z_pval
from sklearn.svm import SVC
import seaborn as sns
from decodanda import plot_perfs_null_model_single

# In a low-dimensional geometry:
#   XOR should not be decodable with a linear decoder
#   XOR should be decodable with a non-linear decoder


def test_non_linear_decoder():
    np.random.seed(0)

    data = generate_synthetic_data(n_neurons=80, n_trials=200, rateB=0.3, rateA=0.3, keyA='stimulus',
                                   keyB='action', mixed_term=0)  # <- Low-D geometry

    XOR = [['11', '00'], ['01', '10']]

    f, ax = plt.subplots(figsize=(4, 3))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Linear SVC', 'Cubic SVC'])
    ax.set_ylabel('XOR Decoding Performance')
    sns.despine(ax=ax)
    ax.set_xlim(-0.5, 1.5)

    # Linear SVC
    d = Decodanda(data=data, conditions={'action': [-1, 1], 'stimulus': [-1, 1]}, verbose=False)
    res, null = d.decode_with_nullmodel(dichotomy=XOR, cross_validations=100, training_fraction=0.8, nshuffles=20, ndata=100)

    assert z_pval(res, null)[1] > 0.05, "XOR should not be decodable with a linear SVC in a low-Dim geometry."

    plot_perfs_null_model_single(res, null, x=0, ax=ax)

    # Cubic SVC
    svc_2 = SVC(C=1.0, kernel='poly', degree=3, gamma=2)
    d = Decodanda(data=data, conditions={'action': [-1, 1], 'stimulus': [-1, 1]}, verbose=False,
                  classifier=svc_2)  # <- using the cubit classifier here
    res, null = d.decode_with_nullmodel(dichotomy=XOR, cross_validations=100, training_fraction=0.8, nshuffles=20, ndata=100)

    assert z_pval(res, null)[1] < 0.05, "XOR should be decodable with a cubic SVC in a low-Dim geometry."

    plot_perfs_null_model_single(res, null, x=1, ax=ax)

    f.savefig('./figures/non_linear_classifier.pdf')


if __name__ == "__main__":
    test_non_linear_decoder()
    print("\nNon-linear Classifier Test Passed")
