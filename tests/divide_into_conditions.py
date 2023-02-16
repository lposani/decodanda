import numpy as np
from decodanda import Decodanda, generate_synthetic_data

# Test the core functionality of the Decodanda class,
#  i.e., dividing the data into conditioned arrays


def test_divide_into_conditions():
    np.random.seed(0)

    # Lots of data: 10,000 time bins divided into 200 trials
    data = generate_synthetic_data(n_neurons=80, n_trials=200, rateB=0.3, rateA=0.3, keyA='stimulus',
                                   keyB='action', timebins_per_trial=50)
    d = Decodanda(data=data, conditions={'action': [-1, 1], 'stimulus': [-1, 1]}, verbose=False)

    assert np.sum([x[0].shape[0] for x in d.conditioned_rasters.values()]) == 10000, "The total number of conditioned data should be 10,000"
    assert np.sum([len(np.unique(x[0])) for x in d.conditioned_trial_index.values()]) == 200, "The total number of trial index should be 200"

    # Sparse data
    data = generate_synthetic_data(n_neurons=80, n_trials=8, rateB=0.3, rateA=0.3, keyA='stimulus',
                                   keyB='action', timebins_per_trial=1)
    d = Decodanda(data=data, conditions={'action': [-1, 1], 'stimulus': [-1, 1]}, verbose=False, min_data_per_condition=0, min_trials_per_condition=0)

    assert np.sum([x[0].shape[0] for x in d.conditioned_rasters.values()]) == 8, "The total number of conditioned data should be 8"
    assert np.sum([len(np.unique(x[0])) for x in d.conditioned_trial_index.values()]) == 8, "The total number of trial index should be 8"


if __name__ == "__main__":
    test_divide_into_conditions()
    print("\nDividing data into conditions - Test Passed")

