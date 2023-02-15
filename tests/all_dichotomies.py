import numpy as np
from decodanda import Decodanda, generate_synthetic_data
from decodanda.utilities import z_pval

# Test the decoding of all dichotomies, including non-balanced ones (not really dichotomies)
# They should be decodable.


def test_balancing_confounds():
    np.random.seed(0)

    data = generate_synthetic_data(n_neurons=80, n_trials=100, rateB=0.3, rateA=0.3, keyA='stimulus', keyB='action', corrAB=0.8)

    d = Decodanda(data=data, conditions={'action': [-1, 1], 'stimulus': [-1, 1]}, verbose=True)

    # Testing all individual dichotomies
    pvals = []
    powerchotomies = d._powerchotomies()

    for key in powerchotomies:
        res, null = d.decode_with_nullmodel(powerchotomies[key],
                                            training_fraction=0.75,
                                            ndata=200,
                                            dic_key=key,
                                            cross_validations=20,
                                            nshuffles=20)
        pvals.append(z_pval(res, null))



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
