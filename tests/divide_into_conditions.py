import numpy as np
from decodanda import Decodanda, generate_synthetic_data
from decodanda.utilities import contiguous_chunking, enforce_min_time_separation

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


def test_divide_into_conditions_with_timeseparation():
    np.random.seed(0)

    # Hand-constructed behavior in chunks (T=20)
    # Two semantic keys, mutually exclusive combinations at each time bin.
    T = 20
    dt = 0.1
    time = np.arange(T) * dt

    # Build 4 chunks across 2 combos, with short gaps in time between chunks
    # (action, stimulus):
    # 0-4   : ( 1,  1)
    # 5-9   : ( 1, -1)
    # 10-14 : ( 1,  1)
    # 15-19 : (-1, -1)
    action = np.array([ 1]*15 + [-1]*5, dtype=float)
    stimulus = np.array([ 1]*5 + [-1]*5 + [1]*5 + [-1]*5, dtype=float)

    # Minimal neural raster, just nonzero so we don't discard everything
    raster = (np.random.rand(T, 5) < 0.2).astype(float)

    session = {
        "raster": raster,
        "action": action,
        "stimulus": stimulus,
        "time": time,
    }

    conditions = {"action": [-1, 1], "stimulus": [-1, 1]}
    min_time_separation = 0.35  # > 0.2s gap between chunk boundaries (dt*2), so should drop some segments

    # Run Decodanda (must be configured to use chunk-based trials)
    d = Decodanda(
        data=session,
        conditions=conditions,
        verbose=False,
        min_data_per_condition=0,
        min_trials_per_condition=0,
        min_activations_per_cell=0,
        neural_attr="raster",
        time_attr="time",
        min_time_separation=min_time_separation,
        trial_attr=None,
        trial_chunk=None,
    )

    # Pull what you now store
    got_trial_vec = d._session_trial_vectors[0]
    got_sep_mask = d._time_separation_masks[0]

    # --- Build expected trial vector exactly from the hand-constructed behavior ---
    total_mask = np.zeros(T, dtype=bool)
    cond_of_bin = np.full(T, -1.0, dtype=float)
    local_trial_of_bin = np.full(T, -1.0, dtype=float)

    combos = [(a, s) for a in conditions["action"] for s in conditions["stimulus"]]
    for ci, (a, s) in enumerate(combos):
        m = (action == a) & (stimulus == s)
        total_mask |= m
        if not np.any(m):
            continue

        local_trials = contiguous_chunking(m)[m].astype(float)

        # mutual exclusivity check
        if np.any(cond_of_bin[m] != -1.0):
            raise ValueError("Test setup error: condition masks overlap.")

        cond_of_bin[m] = float(ci)
        local_trial_of_bin[m] = local_trials

    valid = (cond_of_bin != -1.0) & (local_trial_of_bin != -1.0)
    pairs = np.stack([cond_of_bin[valid], local_trial_of_bin[valid]], axis=1)
    _, inv = np.unique(pairs, axis=0, return_inverse=True)

    exp_trial_vec = np.full(T, -1.0, dtype=float)
    exp_trial_vec[valid] = inv.astype(float)

    exp_sep_mask = enforce_min_time_separation(exp_trial_vec, min_time_separation, time)

    # --- Assertions: internal vectors match expected ---
    assert np.array_equal(got_trial_vec,
                          exp_trial_vec), "session_trial_vector does not match the handcrafted expectation."
    assert np.array_equal(got_sep_mask,
                          exp_sep_mask), "time_separation_mask does not match the handcrafted expectation."


if __name__ == "__main__":
    test_divide_into_conditions()
    test_divide_into_conditions_with_timeseparation()
    print("\nDividing data into conditions - Tests Passed")

