from .decodanda import Decodanda, generate_binary_conditions
from .visualize import line_with_shade
from .imports import *
from .utilities import p_to_ast, z_pval


def time_analysis(data, conditions, time_attr, time_window, decodanda_params, decoding_params, time_boundaries,
                  plot=False):
    """
    :param data: the dataset to be decoded, in the same for as in the Decodanda constructor.
    :param conditions: the variables with values to be decoded, in the same for as in the Decodanda constructor.
    :param time_attr: the variable that defines time from the zero offset.
    :param decodanda_params: dictionary of parameters for the Decodanda constructor.
    :param decoding_params: dictionary of parameters for the Decodanda.decode() function.
    :param time_boundaries: List [min, max]: only trials with data points spanning the whole time
    interval will be considered for the decoding analysis.
    :return: performances, null
    """
    for dataset in data:
        all_times = np.sort(np.unique(dataset[time_attr][np.isnan(dataset[time_attr]) == False]))
        # select only trials that have data up to the time boundaries
        dataset['time_selected'] = np.zeros(len(dataset[time_attr])) * np.nan
        all_times = all_times[all_times >= time_boundaries[0]]
        all_times = all_times[all_times <= time_boundaries[1]]
        for i, t in enumerate(dataset[time_attr]):
            if t == all_times[0]:
                if (dataset[time_attr][i:i + len(all_times)] == all_times).all():
                    dataset['time_selected'][i:i + len(all_times)] = dataset[time_attr][i:i + len(all_times)]
        print("times min: %.2f, max: %.2f - %u trials out of %u" % (
            all_times[0], all_times[-1], np.sum(dataset['time_selected'] == 0), np.sum(dataset[time_attr] == 0)))

    # now assuming time_attr has T unique values that are common for all trials / events

    time_centers = np.linspace(all_times[0], all_times[-1], 1 + floor((all_times[-1] - all_times[0]) / time_window))[
                   :-1]
    performances = {key: np.zeros(len(time_centers)) for key in list(conditions.keys())+['XOR']}
    nulls = {key: np.zeros((len(time_centers), decoding_params['nshuffles'])) for key in list(conditions.keys())+['XOR']}
    pvalues = {key: np.zeros(len(time_centers))*np.nan for key in list(conditions.keys())+['XOR']}

    for i, t in enumerate(time_centers):
        print("\n[Decoding in time]\tdecoding using data in the time window: [%.2f, %.2f]" % (t, t+time_window))
        perfs, null = decoding_in_time(data, conditions, 'time_selected', t, time_window, decodanda_params,
                                       decoding_params)
        for key in perfs:
            performances[key][i] = perfs[key]
            nulls[key][i] = null[key]
            print(key, 'Performance: %.2f' % np.nanmean(perfs[key]), 'Null: %.2f +- %.2f std' %
                  (np.nanmean(null[key]), np.nanstd(null[key])), p_to_ast(z_pval(perfs[key], null[key])[1]))
            pvalues[key][i] = z_pval(perfs[key], null[key])[1]

    if plot:
        nkeys = len(performances.keys())
        xlabels = ['%s\n%s' % (t, t + time_window) for t in time_centers]
        f, ax = plt.subplots(1, nkeys, figsize=(4 * nkeys, 3.5), sharey=True)
        sns.despine(f)
        for i, key in enumerate(list(performances.keys())):
            ax[i].set_xlabel('Time from offset')
            ax[i].set_xticks(time_centers)
            ax[i].set_xticklabels(xlabels)
            ax[0].set_ylabel('Decoding performance')
            ax[i].plot(time_centers, performances[key], linewidth=2, color=pltcolors[i], marker='o')
            line_with_shade(time_centers, nulls[key].T, ax=ax[i], errfunc=lambda x, axis: 2*np.nanstd(x, axis=axis))
            ax[i].set_title(key)
            ax[i].axvline([0], color='k', linestyle='--', alpha=0.5, linewidth=2)
            for t in range(len(time_centers)):
                if pvalues[key][t] < 0.05:
                    ax[i].text(time_centers[t], performances[key][t], p_to_ast(pvalues[key][t]), ha='center', va='bottom', fontsize=11)

    return performances, nulls, time_centers


# Function to decode one specific time bin
def decoding_in_time(data, conditions, time_attr, time, dt, decodanda_params, decoding_params):
    # creating new conditions using the conditions lambda functions plus the specific time filter
    # caution: extreme lambda abstraction involved
    if type(list(conditions.values())[0]) == list:
        conditions = generate_binary_conditions(conditions)
    t_conditions = {}

    for key in conditions:
        t_conditions[key] = {}
        for sub_key in conditions[key]:
            t_conditions[key][sub_key] = lambda d, t0=time, mk=key, k=sub_key: conditions[mk][k](d) & (
                    d[time_attr] >= t0) & (d[
                                               time_attr] < t0 + dt)  # pass these  ^      ^       ^   as default args to allow iteration
    perfs, null = Decodanda(data=data, conditions=t_conditions, **decodanda_params).decode(xor=True, **decoding_params)
    return perfs, null

