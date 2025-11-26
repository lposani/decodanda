import matplotlib.pyplot as plt
import numpy as np
from decodanda import Decodanda, generate_synthetic_data


def _make_multiclass_sessions(
    n_sessions=2,
    n_neurons=5,
    stimulus_values=("A", "B", "C"),   # 3-level variable
    choice_values=("L", "R"),          # 2-level variable
    base_n_bins_per_condition=5,
):
    """
    Build a list of session dicts in the typical Decodanda format:

    session = {
        'raster':   <T x N> array
        'stimulus': <T> array of labels in `stimulus_values`
        'choice':   <T> array of labels in `choice_values`
        'trial':    <T> array of trial IDs
    }

    Each (stimulus, choice) combination gets a *different* number of bins:
    n_bins(combo_k) = base_n_bins_per_condition + k
    (k is the index in the combos list).
    """
    sessions = []
    combos = [(s, c) for s in stimulus_values for c in choice_values]
    n_conditions = len(combos)

    # different n_bins for each combo
    bins_per_combo = {
        combo: base_n_bins_per_condition + k
        for k, combo in enumerate(combos)
    }

    # total bins per session
    T = sum(bins_per_combo[combo] for combo in combos)

    for _ in range(n_sessions):
        # deterministic-ish data: all ones so no neuron is silent
        raster = np.ones((T, n_neurons), dtype=float)

        stim_labels = []
        choice_labels = []
        trial_ids = []

        t = 0
        for combo_idx, (s_val, c_val) in enumerate(combos):
            n_bins = bins_per_combo[(s_val, c_val)]
            for _ in range(n_bins):
                stim_labels.append(s_val)
                choice_labels.append(c_val)
                trial_ids.append(t)  # one "trial" per bin is fine for tests
                t += 1

        session = {
            "raster":   np.asarray(raster),
            "stimulus": np.asarray(stim_labels),
            "choice":   np.asarray(choice_labels),
            "trial":    np.asarray(trial_ids),
        }
        sessions.append(session)

    return sessions, combos, n_neurons, T, bins_per_combo


def test_conditioned_rasters_multiclass_basic():
    """
    Typical multiclass usage:

    - 2 sessions
    - stimulus: 3 values
    - choice:   2 values
    -> 6 condition combinations

    We check that:
    - Decodanda builds the expected number of conditioned rasters
    - Each condition has one array per session
    - Each per-session conditioned array has the expected number of bins
      matching bins_per_combo[(stimulus, choice)]
    - Summing bins across all conditions recovers the original T per session
    """

    sessions, combos, n_neurons, T, bins_per_combo = _make_multiclass_sessions()
    n_sessions = len(sessions)
    n_combos_expected = len(combos)

    stim_vals = [s for s, _ in combos]
    stim_vals_unique = list(dict.fromkeys(stim_vals))
    choice_vals = [c for _, c in combos]
    choice_vals_unique = list(dict.fromkeys(choice_vals))

    conditions = {
        "stimulus": stim_vals_unique,
        "choice": choice_vals_unique,
    }

    dec = Decodanda(
        data=sessions,
        conditions=conditions,
        classifier="svc",
        min_data_per_condition=1,
        min_trials_per_condition=1,
        squeeze_trials=False,
        exclude_silent=False,
        verbose=True,
    )

    # 1) number of condition combinations
    assert isinstance(dec.conditioned_rasters, dict)
    assert len(dec.conditioned_rasters) == n_combos_expected

    # 2–3) each condition key: one array per session, with the right number of bins
    for key, per_session_list in dec.conditioned_rasters.items():
        assert isinstance(per_session_list, list)
        assert len(per_session_list) == n_sessions

        # recover (stimulus, choice) from semantic vector string "(A L)"
        sem = dec._semantic_vectors[key]
        inner = sem.strip()[1:-1]      # remove parentheses
        tokens = inner.split()
        assert len(tokens) == 2        # stimulus, choice
        combo = (tokens[0], tokens[1])

        expected_bins = bins_per_combo[combo]

        for arr in per_session_list:
            assert arr.ndim == 2
            assert arr.shape[1] == n_neurons
            assert arr.shape[0] == expected_bins

    # 4) per-session coverage: total bins across conditions == original T
    for session_idx in range(n_sessions):
        total_bins = 0
        for arr_list in dec.conditioned_rasters.values():
            total_bins += arr_list[session_idx].shape[0]
        assert total_bins == T


# test_decodanda_multiclass_decoding.py


def _make_fake_multiclass_data(
    n_neurons=20,
    T=6000,
    n_values1=3,
    n_values2=3,
    snr1=0.0,
    snr2=0.0,
    corr=0.0,
    seed=0,
):
    """
    Generate a single-session dataset with two discrete variables (var1, var2).

    - var1 takes n_values1 values: 0,1,...,n_values1-1
    - var2 takes n_values2 values: 0,1,...,n_values2-1
    - Each neuron fires according to:
          r_t ~ N( snr1 * a[var1_t] + snr2 * b[var2_t], I )
      where a and b are random prototype vectors (one per value).
    - (var1_t, var2_t) are drawn from a correlated joint distribution:
          P(i,j) ∝ 1 + corr * (i == j)
      then normalized.

    Returns
    -------
        sessions : list with a single Decodanda-style session dict
        (var1_vals, var2_vals) : the arrays of labels for this session
    """
    rng = np.random.RandomState(seed)

    # prototypes for each value of each variable
    a = rng.randn(n_values1, n_neurons)
    b = rng.randn(n_values2, n_neurons)

    # joint probability over (var1, var2)
    P = np.ones((n_values1, n_values2))
    for i in range(min(n_values1, n_values2)):
        P[i, i] += corr * max(n_values1, n_values2)
    P = P / P.sum()
    P_flat = P.ravel()

    idx_flat = rng.choice(n_values1 * n_values2, size=T, p=P_flat)
    var1_vals = idx_flat // n_values2
    var2_vals = idx_flat % n_values2

    raster = np.zeros((T, n_neurons), dtype=float)
    for t, (i, j) in enumerate(zip(var1_vals, var2_vals)):
        mu = snr1 * a[i] + snr2 * b[j]
        noise = rng.randn(n_neurons)
        raster[t, :] = mu + noise

    session = {
        "raster": raster,
        "var1": var1_vals,
        "var2": var2_vals,
        "trial": np.arange(T),  # one "trial" per bin
    }

    return [session], (var1_vals, var2_vals)


# ---------------------------------------------------------------------
# Tests using decode_multiclass_with_nullmodel
# ---------------------------------------------------------------------


def test_multiclass_high_snr_balanced_decoding_vs_null():
    """
    var1 has high SNR, var2 irrelevant.
    With balanced conditions (var1,var2), decoding var1 should be clearly
    above its null model.
    """
    sessions, _ = _make_fake_multiclass_data(
        n_neurons=20,
        T=6000,
        n_values1=3,
        n_values2=3,
        snr1=1.5,   # strong encoding of var1
        snr2=0.0,
        corr=0.0,
        seed=1,
    )

    conditions = {
        "var1": [0, 1, 2],
        "var2": [0, 1, 2],
    }

    dec = Decodanda(
        data=sessions,
        conditions=conditions,
        classifier="svc",
        min_data_per_condition=1,
        min_trials_per_condition=1,
        squeeze_trials=False,
        exclude_silent=False,
        verbose=True,
    )

    perf, perf_null, cm, cm_null = dec.decode_multiclass_with_nullmodel(
        variable="var1",
        training_fraction=0.5,
        cross_validations=5,
        nshuffles=10,
        ndata=50,
        subsample=0,
        plot=True
    )
    z = (perf - np.nanmean(perf_null)) / np.nanstd(perf_null)
    # we just require true performance to be clearly above null
    assert z > 3, (
        f"High-SNR var1 decoding not sufficiently above null: "
        f"perf={perf:.2f}, null_mean={np.nanmean(perf_null):.2f}, null_std={np.nanstd(perf_null):.2f} - z={z:.2f}"
    )


def test_multiclass_snr0_correlated_unbalanced_vs_null():
    """
    var1 has SNR=0, var2 high SNR, strong correlation between var1 and var2.

    If we *only* give Decodanda var1 as a condition (unbalanced with respect to var2),
    decoding var1 should be above its null model because the decoder can exploit
    the correlation with var2.
    """
    sessions, _ = _make_fake_multiclass_data(
        n_neurons=20,
        T=1000,
        n_values1=3,
        n_values2=3,
        snr1=0.0,   # var1 not encoded directly
        snr2=3.0,   # var2 strongly encoded
        corr=0.9,   # strong statistical correlation
        seed=2,
    )

    # UNBALANCED: only var1 is listed in conditions
    conditions_unbalanced = {
        "var1": [0, 1, 2],
    }

    dec_unbalanced = Decodanda(
        data=sessions,
        conditions=conditions_unbalanced,
        classifier="svc",
        min_data_per_condition=1,
        min_trials_per_condition=1,
        squeeze_trials=False,
        exclude_silent=False,
        verbose=False,
    )

    perf, perf_null, cm, cm_null = dec_unbalanced.decode_multiclass_with_nullmodel(
        variable="var1",
        training_fraction=0.5,
        cross_validations=5,
        nshuffles=10,
        ndata=80,
        subsample=0,
        plot=True
    )

    z = (perf - np.nanmean(perf_null)) / np.nanstd(perf_null)
    # because of correlation with high-SNR var2, performance should exceed null
    assert z > 2.5, (
        f"Unbalanced decoding with correlated high-SNR var2 not clearly above null"
        f"perf={perf:.2f}, null_mean={np.nanmean(perf_null):.2f}, null_std={np.nanstd(perf_null):.2f} - z={z:.2f}"

    )


def test_multiclass_snr0_correlated_balanced_matches_null():
    """
    Same dataset as above: var1 SNR=0, var2 high SNR, strong correlation.

    With balanced conditions (var1,var2), decoding var1 should *not* be better
    than its null model, because the decoder can no longer exploit the
    correlation with var2.
    """
    sessions, _ = _make_fake_multiclass_data(
        n_neurons=20,
        T=1000,
        n_values1=3,
        n_values2=3,
        snr1=0.0,
        snr2=3.0,
        corr=0.9,
        seed=3,
    )

    # BALANCED: both variables in conditions
    conditions_balanced = {
        "var1": [0, 1, 2],
        "var2": [0, 1, 2],
    }

    dec_balanced = Decodanda(
        data=sessions,
        conditions=conditions_balanced,
        classifier="svc",
        min_data_per_condition=1,
        min_trials_per_condition=1,
        squeeze_trials=False,
        exclude_silent=False,
        verbose=True,
    )

    dec_balanced.decode(
        training_fraction=0.5,
        cross_validations=5,
        nshuffles=10,
        ndata=80,
        subsample=0,
        plot_all=True,
        plot=True
    )

    perf, perf_null, cm, cm_null = dec_balanced.decode_multiclass_with_nullmodel(
        variable="var1",
        training_fraction=0.5,
        cross_validations=5,
        nshuffles=10,
        ndata=80,
        subsample=0,
        plot=True
    )
    z = (perf - np.nanmean(perf_null)) / np.nanstd(perf_null)
    # because of correlation with high-SNR var2, performance should exceed null
    assert z < 2, (
        f"Balanced decoding with SNR=0 should match null; "
        f"perf={perf:.2f}, null_mean={np.nanmean(perf_null):.2f}, null_std={np.nanstd(perf_null):.2f} - z={z:.2f}"

    )


if __name__ == "__main__":
    test_conditioned_rasters_multiclass_basic()
    test_multiclass_high_snr_balanced_decoding_vs_null()
    test_multiclass_snr0_correlated_unbalanced_vs_null()
    test_multiclass_snr0_correlated_balanced_matches_null()
    print("\nMulticlass Test Passed")

