import numpy as np
from decodanda import Decodanda, generate_synthetic_data
from decodanda.utilities import z_pval

# Test disentangling confounds: the "action" variable should be decodable when analyzed
# individually but NOT be decodable when properly balanced with the "stimulus" one.


def test_balancing_confounds():
    np.random.seed(0)

    data = generate_synthetic_data(n_neurons=80, n_trials=100, rateB=0, rateA=0.3, keyA='stimulus', keyB='action', corrAB=0.8)

    # This analysis should give a positive performance
    res_unbalanced, null_unbalanced = Decodanda(
        data=data,
        conditions={'action': [-1, 1]}
        ).decode(training_fraction=0.75)

    # This analysis should give a negative performance
    res_balanced, null_balanced = Decodanda(
        data=data,
        conditions={'action': [-1, 1], 'stimulus': [-1, 1]}
        ).decode(training_fraction=0.75)

    zb, _ = z_pval(res_balanced['action'], null_balanced['action'])
    zu, _ = z_pval(res_unbalanced['action'], null_unbalanced['action'])

    assert zu > 2.0, "Should be significant"
    assert zb < 2.0, "Should not be significant"


if __name__ == "__main__":
    test_balancing_confounds()
    print("\nBalancing cofounds test passed")