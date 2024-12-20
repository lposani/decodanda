import numpy as np
import pytest
from decodanda import Decodanda


@pytest.mark.parametrize("data", indirect=True, argvalues=["base_dataset",
                                                           "correlated_dataset",
                                                           "undersampled_dataset",
                                                           "random_dataset"])
def test_division_into_conditions(data, initialization_parameters):
    """
    Test the division of the dataset into conditions. First, a decodanda object is created using the dataset as input.
    Next, we check that the number of features is the same for all conditioned data. Then, we check that the number of
    we check that the number of features, samples and trials in the decodanda object is less than or equal to the
    number of features, samples and trials in the dataset.

    Parameters
    ----------
    data: Dataset
        The dataset to test
    """
    decodanda = Decodanda(data=dict(data.data),
                          conditions=dict(data.conditions),
                          **initialization_parameters)
    max_features_expected = data.num_neurons
    max_samples_expected = data.num_samples
    num_features = {raster[0].shape[1] for raster in decodanda.conditioned_rasters.values()}
    if len(num_features) > 1:
        raise AssertionError("The number of features should be the same for all conditioned data")
    num_features = num_features.pop()
    num_samples = sum(raster[0].shape[0] for raster in decodanda.conditioned_rasters.values())
    try:
        assert max_features_expected >= num_features
        assert max_samples_expected >= num_samples
    except AssertionError as exc:
        raise AssertionError("The concatenated shape of conditioned data should be"
                             "less than or equal to the number of features x samples in the dataset") from exc
    max_trials_expected = data.num_trials
    num_trials = len(np.concatenate([np.unique(trial) for trials in decodanda.conditioned_trial_index.values()
                                     for trial in trials]))
    assert (max_trials_expected >= num_trials), \
        "The total number of trial index should be less than or equal to the number of trials in the dataset"
