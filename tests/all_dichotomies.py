import matplotlib.pyplot as plt
import numpy as np
from decodanda import Decodanda, generate_synthetic_data
from decodanda.utilities import z_pval
from tqdm import tqdm
from decodanda import plot_perfs_null_model_single
# Test the decoding of all dichotomies, including non-balanced ones (not really dichotomies)
# They should be decodable.


def teat_all_dichotomies():
    np.random.seed(0)

    data = generate_synthetic_data(n_neurons=80, n_trials=100, rateB=0.3, rateA=0.3, keyA='stimulus',
                                   keyB='action', mixed_term=0)

    d = Decodanda(data=data, conditions={'action': [-1, 1], 'stimulus': [-1, 1]}, verbose=False)

    # Testing all individual dichotomies
    pvals = []
    powerchotomies = d._powerchotomies()

    x=0
    f, ax = plt.subplots(figsize=(6, 4))
    for key in tqdm(list(powerchotomies.keys())):
        res, null = d.decode_with_nullmodel(powerchotomies[key],
                                            training_fraction=0.75,
                                            ndata=50,
                                            dic_key=key,
                                            cross_validations=10,
                                            nshuffles=10)
        pvals.append(z_pval(res, null))
        plot_perfs_null_model_single(res, null, x=x, ax=ax, marker='o', color='k')
        x += 1
    ax.set_xticks(np.arange(len(powerchotomies)))
    ax.set_xticklabels(list(powerchotomies.keys()), rotation=60)

    assert np.max(pvals[:-1]) < 0.05, "All non-XOR dichotomies should be decodable"
    assert pvals[-1] > 0.05, "XOR should not be decodable"

if __name__ == "__main__":
    teat_all_dichotomies()
    print("\nAll Decodable Dichotomies Test Passed")
