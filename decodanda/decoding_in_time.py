from decodanda import Decodanda
from imports import *


def time_analysis(data, conditions, time_attr, decodanda_params, decoding_params, time_boundaries=None):
    """
    :param data: the dataset to be decoded, in the same for as in the Decodanda constructor.
    :param conditions: the variables with values to be decoded, in the same for as in the Decodanda constructor.
    :param time_attr: the variable that defines time from the zero offset.
    :param decodanda_params: dictionary of parameters for the Decodanda constructor.
    :param decoding_params: dictionary of parameters for the Decodanda.decode() function.
    :param time_boundaries: if None, they will be set automatically as those minimum and maximum time such that
    there are as many data vector as time=0. If specified, only trials with data points spanning the whole time
    interval will be considered for the decoding analysis.
    :return: performances, null
    """
    all_times = np.sort(np.unique(data[time_attr][np.isnan(data[time_attr]) == False]))
    min_time_index = np.where(all_times == 0)[0][0]
    max_time_index = np.where(all_times == 0)[0][0]

    if time_boundaries is None:
        # find minimum and maximum such that we have the same number of data for each time point
        ntrials = np.sum(data[time_attr] == 0)
        while (np.sum(data[time_attr] == all_times[min_time_index - 1])) == ntrials:
            min_time_index -= 1
        while (np.sum(data[time_attr] == all_times[max_time_index])) == ntrials:
            max_time_index += 1
        print("Found min: %u, max: %u" % (all_times[min_time_index], all_times[max_time_index]))
        all_times = all_times[min_time_index:max_time_index]
    else:
        # select only trials that have data up to the time boundaries
        data['time_selected'] = np.zeros(len(data[time_attr])) * np.nan
        all_times = all_times[all_times >= time_boundaries[0]]
        all_times = all_times[all_times <= time_boundaries[1]]
        for i, t in enumerate(data[time_attr]):
            if t == all_times[0]:
                print(data[time_attr][i:i+len(all_times)])
                if (data[time_attr][i:i+len(all_times)] == all_times).all():
                    data['time_selected'][i:i+len(all_times)] = data[time_attr][i:i+len(all_times)]
        time_attr = 'time_selected'

    # assuming time_attr has T unique values that are common for all trials / events
    performances = {key: np.zeros(len(all_times)) for key in conditions}
    nulls = {key: np.zeros((len(all_times), decoding_params['nshuffles'])) for key in conditions}
    for i, t in enumerate(all_times):
        perfs, null = decoding_in_time(data, conditions, time_attr, t, decodanda_params, decoding_params)
        for key in perfs:
            performances[key][i] = perfs[key]
            nulls[key][i] = null[key]
    return performances, nulls


def decoding_in_time(data, conditions, time_attr, time, decodanda_params, decoding_params):
    # creating new conditions using the conditions lambda functions plus the specific time filter
    # caution: extreme lambda abstraction involved
    t_conditions = {}
    for key in conditions:
        t_conditions[key] = {}
        for sub_key in conditions[key]:
            t_conditions[key][sub_key] = lambda d, t0=time, k=sub_key: conditions[key][k](d) & (
                    d[time_attr] == t0)  # pass these  ^      ^   as default args to allow iteration
    perfs, null = Decodanda(data=data, conditions=t_conditions, **decodanda_params).decode(**decoding_params)
    return perfs, null
