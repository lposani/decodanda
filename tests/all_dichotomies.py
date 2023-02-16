import matplotlib.pyplot as plt
import numpy as np
from decodanda import Decodanda, generate_synthetic_data
from decodanda.utilities import z_pval
from tqdm import tqdm
from decodanda import plot_perfs_null_model_single
import seaborn as sns

# Test the decoding of all dichotomies, including non-balanced ones
# All dichotomies should be decodable in a high-dimensional geometry
# All dichotomies except XOR should be decodable in a low-dimensional geometry


def test_all_dichotomies():
    np.random.seed(0)

    data = generate_synthetic_data(n_neurons=80, n_trials=100, rateB=0.3, rateA=0.3, keyA='stimulus',
                                   keyB='action', mixed_term=0)

    d = Decodanda(data=data, conditions={'action': [-1, 1], 'stimulus': [-1, 1]}, verbose=False)

    # Testing all individual dichotomies - low dimensional representations
    pvals = []
    powerchotomies = d._powerchotomies()

    x = 0
    f, ax = plt.subplots(figsize=(6, 4))
    for key in tqdm(list(powerchotomies.keys())):
        res, null = d.decode_with_nullmodel(powerchotomies[key],
                                            training_fraction=0.9,
                                            ndata=100,
                                            dic_key=key,
                                            cross_validations=20,
                                            nshuffles=20)
        pvals.append(z_pval(res, null)[1])
        plot_perfs_null_model_single(res, null, x=x, ax=ax, marker='o', color='k')
        x += 1
    ax.set_xticks(np.arange(len(powerchotomies)))
    ax.set_xticklabels(list(powerchotomies.keys()), rotation=60)
    sns.despine(f)
    ax.set_ylabel('Decoding Performance')
    ax.set_xlabel('Dichotomy')
    ax.set_title('Low-D Geometry')
    f.savefig('./figures/all_dichotomies_lowd.pdf')
    assert np.max(pvals[:-1]) < 0.05, "All non-XOR dichotomies should be decodable in a Low-D geometry."
    assert pvals[-1] > 0.05, "XOR should not be decodable in a low-D geometry."


if __name__ == "__main__":
    test_all_dichotomies()
    print("\nAll Decodable Dichotomies Test Passed")
