from decodanda.visualize import *
from decodanda.utilities import *
from decodanda.imports import *
from decodanda.classes import Decodanda


# Single neurons analysis functions

def ablation_analysis(self, percentiles, cross_validations=25, training_fraction=0.9, metric='weight',
                      plot=False, axs=None, ndata='auto'):
    if metric == 'weight':
        self.decoding_weights = {key: [] for key in self._semantic_keys}
        self.semantic_decode(training_fraction=0.9, cross_validations=cross_validations, nshuffles=0)
        mean_decoding_weights = {key: np.abs(np.nanmean(self.decoding_weights[key], 0))[0] for key in
                                 self._semantic_keys}

    key1 = self._semantic_keys[0]
    key2 = self._semantic_keys[1]

    if metric == 'weight':
        sel1 = np.abs(mean_decoding_weights[key1]) / np.sqrt(np.sum(mean_decoding_weights[key1] ** 2))
        sel2 = np.abs(mean_decoding_weights[key2]) / np.sqrt(np.sum(mean_decoding_weights[key2] ** 2))

    if metric == 'selectivity':
        rate_i_1 = np.nanmean([self.centroids['10'], self.centroids['11']], 0)
        rate_i_2 = np.nanmean([self.centroids['00'], self.centroids['01']], 0)
        sel1 = np.abs(rate_i_1 - rate_i_2)

        rate_j_1 = np.nanmean([self.centroids['01'], self.centroids['11']], 0)
        rate_j_2 = np.nanmean([self.centroids['00'], self.centroids['10']], 0)
        sel2 = np.abs(rate_j_1 - rate_j_2)

    specialization = np.abs(sel1 - sel2) / np.abs(sel1 + sel2)
    information = np.abs(sel1 + sel2)

    data_spec = {
        'Decoding': {
            key1: [],
            key2: []
        },
        'CCGP': {
            key1: [],
            key2: []
        }
    }

    data_info = {
        'Decoding': {
            key1: [],
            key2: []
        },
        'CCGP': {
            key1: [],
            key2: []
        }
    }

    if ndata == 'auto':
        ndata = 2 * len(self.subset)

    for i, sel_perc in tqdm(enumerate(percentiles)):
        # ablation by specialization
        ablated_cells = specialization <= np.percentile(specialization, 100 - sel_perc)
        self.subset = np.where(ablated_cells)[0]
        res_spec, _ = self.semantic_decode(training_fraction=training_fraction, cross_validations=cross_validations,
                                           nshuffles=0, ndata=ndata)

        ccg_spec, _ = self.semantic_CCGP(ntrials=5, nshuffles=0, ndata=ndata)
        self._reset_random_subset()

        data_spec['Decoding'][key1].append(res_spec[key1])
        data_spec['Decoding'][key2].append(res_spec[key2])

        data_spec['CCGP'][key1].append(ccg_spec[key1])
        data_spec['CCGP'][key2].append(ccg_spec[key2])

        # ablation by information
        ablated_cells = information <= np.percentile(information, 100 - sel_perc)
        self.subset = np.where(ablated_cells)[0]
        res_info, _ = self.semantic_decode(training_fraction=training_fraction, cross_validations=cross_validations,
                                           nshuffles=0, ndata=ndata)
        ccg_info, _ = self.semantic_CCGP(ntrials=5, nshuffles=0, ndata=ndata)
        self._reset_random_subset()

        data_info['Decoding'][key1].append(res_info[key1])
        data_info['Decoding'][key2].append(res_info[key2])

        data_info['CCGP'][key1].append(ccg_info[key1])
        data_info['CCGP'][key2].append(ccg_info[key2])

    if plot or axs is not None:
        if axs is None:
            f, axs = plt.subplots(1, 4, figsize=(18, 4), gridspec_kw={'width_ratios': [2.5, 2.5, 2, 2]})
            sns.despine(f)
        axs[0].set_title('Decoding Performance')
        axs[0].set_xlabel('Top % ablated neurons')
        axs[0].set_ylabel('Decoding Performance')

        axs[1].set_title('CCGP')
        axs[1].set_xlabel('Top % ablated neurons')
        axs[1].set_ylabel('CCGP')

        axs[2].set_xlabel('%s %s' % (metric, key1))
        axs[2].set_ylabel('%s %s' % (metric, key2))
        axs[3].set_xlabel('Information')
        axs[3].set_ylabel('Specialization')

        axs[0].plot(percentiles, data_spec['Decoding'][key1], color=pltcolors[0], linewidth=2, alpha=0.7,
                    label=key1 + ' Spec')
        axs[0].plot(percentiles, data_spec['Decoding'][key2], color=pltcolors[1], linewidth=2, alpha=0.7,
                    label=key2 + ' Spec')
        axs[0].plot(percentiles, data_info['Decoding'][key1], color=pltcolors[0], linewidth=2, alpha=0.7,
                    linestyle='--', label=key1 + ' Info')
        axs[0].plot(percentiles, data_info['Decoding'][key2], color=pltcolors[1], linewidth=2, alpha=0.7,
                    linestyle='--', label=key2 + ' Info')

        axs[1].plot(percentiles, data_spec['CCGP'][key1], color=pltcolors[0], linewidth=2, alpha=0.7,
                    label=key1 + ' Spec')
        axs[1].plot(percentiles, data_spec['CCGP'][key2], color=pltcolors[1], linewidth=2, alpha=0.7,
                    label=key2 + ' Spec')
        axs[1].plot(percentiles, data_info['CCGP'][key1], color=pltcolors[0], linewidth=2, alpha=0.7,
                    linestyle='--', label=key1 + ' Info')
        axs[1].plot(percentiles, data_info['CCGP'][key2], color=pltcolors[1], linewidth=2, alpha=0.7,
                    linestyle='--', label=key2 + ' Info')

        corr_scat_kde(sel1, sel2, ax=axs[2], alpha=0.5, s=10)
        corr_scat_kde(information, specialization, ax=axs[3], alpha=0.5, s=10)

        axs[0].legend()

    return data_spec, data_info, sel1, sel2, specialization, information


def single_neuron_modulation(self, nshuffles=2000, normalized=False, plot=False, axs=None):
    rate_i_1 = np.nanmean([self.centroids['10'], self.centroids['11']], 0)
    rate_i_2 = np.nanmean([self.centroids['00'], self.centroids['01']], 0)
    modulation_1 = np.abs(rate_i_1 - rate_i_2)

    rate_j_1 = np.nanmean([self.centroids['01'], self.centroids['11']], 0)
    rate_j_2 = np.nanmean([self.centroids['00'], self.centroids['10']], 0)
    modulation_2 = np.abs(rate_j_1 - rate_j_2)
    if normalized:
        modulation_1 = modulation_1 / (rate_i_1 + rate_i_2)
        modulation_2 = modulation_2 / (rate_j_1 + rate_j_2)

    null_1 = np.zeros((nshuffles, len(modulation_1)))
    null_2 = np.zeros((nshuffles, len(modulation_2)))

    for i in range(nshuffles):
        self._shuffle_conditioned_arrays([['10', '01'], ['11', '00']])
        self._compute_centroids()
        rate_i_1 = np.nanmean([self.centroids['10'], self.centroids['11']], 0)
        rate_i_2 = np.nanmean([self.centroids['00'], self.centroids['01']], 0)
        null_1[i, :] = np.abs(rate_i_1 - rate_i_2)

        rate_j_1 = np.nanmean([self.centroids['01'], self.centroids['11']], 0)
        rate_j_2 = np.nanmean([self.centroids['00'], self.centroids['10']], 0)
        null_2[i, :] = np.abs(rate_j_1 - rate_j_2)

        if normalized:
            null_1[i, :] = null_1[i, :] / (rate_i_1 + rate_i_2)
            null_2[i, :] = null_2[i, :] / (rate_j_1 + rate_j_2)

        self._order_conditioned_rasters()
        self._compute_centroids()

    z1 = (modulation_1 - np.nanmean(null_1, 0)) / np.nanstd(null_1, 0)
    z2 = (modulation_2 - np.nanmean(null_2, 0)) / np.nanstd(null_2, 0)

    selective_1 = (z1 > 2.0) & (z2 < 2.0)
    selective_2 = (z1 < 2.0) & (z2 > 2.0)
    selective_mix = (z1 > 2.0) & (z2 > 2.0)
    non_selective = (z1 < 2.0) & (z2 < 2.0)

    if plot:
        if axs is None:
            f, axs = plt.subplots(1, 2, figsize=(6, 4), gridspec_kw={'width_ratios': [3, 1]})
            sns.despine(f)
        axs[0].axhline([0], color='k')
        axs[0].axvline([0], color='k')
        axs[0].set_xlabel('Firing rate modulation (%s)' % self._semantic_keys[0].split(' ')[0])
        axs[0].set_ylabel('Firing rate modulation (%s)' % self._semantic_keys[1].split(' ')[0])

        axs[0].scatter(modulation_1[non_selective], modulation_2[non_selective], alpha=0.3, color='k',
                       label='Non selective', marker='x')
        axs[0].scatter(modulation_1[selective_1], modulation_2[selective_1], alpha=0.5,
                       label=self._semantic_keys[0].split(' ')[0], marker='o')
        axs[0].scatter(modulation_1[selective_2], modulation_2[selective_2], alpha=0.5,
                       label=self._semantic_keys[1].split(' ')[0], marker='o')
        axs[0].scatter(modulation_1[selective_mix], modulation_2[selective_mix], alpha=0.5, label='Both', marker='o')
        axs[0].legend()
        axs[1].bar([0], np.nanmean(selective_1))
        axs[1].bar([1], np.nanmean(selective_2))
        axs[1].bar([2], np.nanmean(selective_mix))
        axs[1].bar([3], np.nanmean(non_selective), color='k', alpha=0.5)
        axs[1].set_xticks([0, 1, 2, 3])
        axs[1].set_xticklabels(
            [self._semantic_keys[0].split(' ')[0], self._semantic_keys[1].split(' ')[0], 'Both', 'None'], rotation=60)
        axs[1].set_ylabel('Fraction of modulated neurons')

    return selective_1, selective_2, selective_mix, non_selective


def population_selectivity(self, nshuffles=100, min_t=0.5, max_t=0.95, n_t=500, plot=True, axs=None,
                           metric='weight'):
    if metric == 'weight':
        if len(self.decoding_weights[self._semantic_keys[0]]) == 0:
            self.semantic_decode(training_fraction=0.9, nshuffles=0)

        mean_decoding_weights = {key: np.abs(np.nanmean(self.decoding_weights[key], 0))[0] for key in
                                 self._semantic_keys}

    thresholds = np.linspace(min_t, max_t, n_t)
    sd = self._find_semantic_dichotomies()[0]

    data = {}
    null = {}

    for i in range(len(self._semantic_keys)):
        for j in range(i + 1, len(self._semantic_keys)):
            key1 = self._semantic_keys[i]
            key2 = self._semantic_keys[j]

            if metric == 'weight':
                data1 = mean_decoding_weights[key1] / np.sqrt(np.sum(mean_decoding_weights[key1] ** 2.))
                data2 = mean_decoding_weights[key2] / np.sqrt(np.sum(mean_decoding_weights[key2] ** 2.))

            if metric == 'rate':
                rate_i_1 = np.nanmean([self.centroids[k] for k in sd[i][0]], 0)
                rate_i_2 = np.nanmean([self.centroids[k] for k in sd[i][1]], 0)
                data1 = np.abs(rate_i_1 - rate_i_2)

                rate_j_1 = np.nanmean([self.centroids[k] for k in sd[j][0]], 0)
                rate_j_2 = np.nanmean([self.centroids[k] for k in sd[j][1]], 0)
                data2 = np.abs(rate_j_1 - rate_j_2)

            selectivity_data = (data1 - data2) / (data1 + data2)
            angle_data = np.arctan2(data2, data1)
            lengths = np.sqrt(data1 ** 2 + data2 ** 2)

            fractions_var1 = np.asarray([np.mean(selectivity_data > t) for t in thresholds])
            fractions_var2 = np.asarray([np.mean(selectivity_data < -t) for t in thresholds])

            fractions_null1 = np.zeros((n_t, nshuffles))
            fractions_null2 = np.zeros((n_t, nshuffles))

            angles_null = np.zeros((len(lengths), nshuffles))

            for n in range(nshuffles):
                angle_null = np.random.rand(len(lengths)) * np.pi / 2
                data1_null = np.cos(angle_null) * lengths
                data2_null = np.sin(angle_null) * lengths
                selectivity_null = (data1_null - data2_null) / (data1_null + data2_null)
                fractions_null1[:, n] = np.asarray([np.mean(selectivity_null > t) for t in thresholds])
                fractions_null2[:, n] = np.asarray([np.mean(selectivity_null < -t) for t in thresholds])

                angles_null[:, n] = angle_null

            if plot:
                if axs is None:
                    f, axs = plt.subplots(1, 3, figsize=(9, 3.5))

                # first plot: scatter of decoding weights
                for ni in range(self.n_brains):
                    this_brain = self.which_brain == ni + 1
                    axs[0].scatter(data1[this_brain], data2[this_brain], alpha=0.5, color=pltcolors[ni],
                                   label='Brain %u' % (ni + 1), marker='.')
                    if metric == 'weight':
                        axs[0].set_xlabel('Decoding weights (%s)' % key1)
                        axs[0].set_ylabel('Decoding weights (%s)' % key2)
                    if metric == 'rate':
                        axs[0].set_xlabel('Absolute rate selectivity (%s)' % key1)
                        axs[0].set_ylabel('Absolute rate selectivity (%s)' % key2)

                axs[0].axhline([0], color='k')
                axs[0].axvline([0], color='k')
                [xm, xM] = axs[0].get_xlim()
                [ym, yM] = axs[0].get_ylim()
                axs[0].set_xlim([np.min([xm, ym]), np.max([xM, yM])])
                axs[0].set_ylim([np.min([xm, ym]), np.max([xM, yM])])

                # second plot: angle distribution
                axs[1].hist(angle_data, bins=min(max(10, int(len(data1) / 16)), 21), alpha=0.7, label='Data',
                            density=True)
                axs[1].hist(angles_null.flatten(), color='k', histtype='step', label='Null model', alpha=0.5,
                            density=True)
                axs[1].set_xlabel('Angle $\\theta$')
                axs[1].set_ylabel('Count')

                # # third plot: comparison to null model at different selectivity thresholds
                # mean_null = np.nanmean(fractions_null, axis=1)
                # perc_95_null = np.percentile(fractions_null, 95, axis=1)
                # perc_99_null = np.percentile(fractions_null, 99, axis=1)
                # perc_5_null = np.percentile(fractions_null, 5, axis=1)
                # perc_1_null = np.percentile(fractions_null, 1, axis=1)
                #
                # axs[2].set_xlabel('Selectivity threshold')
                # axs[2].set_ylabel('% of selective cells')
                #
                # axs[2].plot(thresholds, mean_null, color='k', label='Null model', linewidth=1.5)
                # # axs[2].plot(thresholds, perc_95_null, color='k', linestyle='-', alpha=0.5)
                # # axs[2].plot(thresholds, perc_5_null, color='k', linestyle='-', alpha=0.5)
                # axs[2].fill_between(thresholds, perc_5_null, perc_95_null, color='k', alpha=0.2)
                # axs[2].fill_between(thresholds, perc_1_null, perc_5_null, color='k', alpha=0.1)
                # axs[2].fill_between(thresholds, perc_95_null, perc_99_null, color='k', alpha=0.1)
                #
                # axs[2].plot(thresholds, fractions, color=pltcolors[0], linewidth=1.5, label='Data')
                # axs[2].legend()

                plot_data = {
                    key1: np.nanmean(fractions_var1),
                    key2: np.nanmean(fractions_var2)
                }
                plot_null = {
                    key1: np.nanmean(fractions_null1, axis=0),
                    key2: np.nanmean(fractions_null2, axis=0)
                }
                plot_perfs_null_model(plot_data, plot_null, ylabel='Mean Selectivity', ax=axs[2],
                                      shownull='violin', setup=False)

                sns.despine(ax=axs[0])
                sns.despine(ax=axs[1])
                sns.despine(ax=axs[2])
                data['%s-%s' % (key1, key2)] = plot_data
                null['%s-%s' % (key1, key2)] = plot_null

    return data, null


def selectivity_angle(self, nshuffles=100, angles=5, plot=True, axs=None, metric='weight', cross_validations=20):
    if metric == 'weight':
        if len(self.decoding_weights[self._semantic_keys[0]]) == 0:
            self.semantic_decode(training_fraction=0.9, nshuffles=0, cross_validations=cross_validations)

        mean_decoding_weights = {key: np.abs(np.nanmean(self.decoding_weights[key], 0))[0] for key in
                                 self._semantic_keys}

    thresholds = np.linspace(0, np.pi / 2, angles + 1)
    dx = thresholds[1] - thresholds[0]
    sd = self._find_semantic_dichotomies()[0]

    for i in range(len(self._semantic_keys)):
        for j in range(i + 1, len(self._semantic_keys)):
            key1 = self._semantic_keys[i]
            key2 = self._semantic_keys[j]

            if metric == 'weight':
                data1 = mean_decoding_weights[key1] / np.sqrt(np.sum(mean_decoding_weights[key1] ** 2.))
                data2 = mean_decoding_weights[key2] / np.sqrt(np.sum(mean_decoding_weights[key2] ** 2.))

            if metric == 'rate':
                rate_i_1 = np.nanmean([self.centroids[k] for k in sd[i][0]], 0)
                rate_i_2 = np.nanmean([self.centroids[k] for k in sd[i][1]], 0)
                data1 = np.abs(rate_i_1 - rate_i_2)

                rate_j_1 = np.nanmean([self.centroids[k] for k in sd[j][0]], 0)
                rate_j_2 = np.nanmean([self.centroids[k] for k in sd[j][1]], 0)
                data2 = np.abs(rate_j_1 - rate_j_2)

            angle_data = np.arctan2(data2, data1)
            lengths = np.sqrt(data1 ** 2 + data2 ** 2)

            fractions, bins = np.histogram(angle_data, thresholds)
            fractions = fractions / np.sum(fractions)
            fractions_null = np.zeros((angles, nshuffles))

            angles_null = np.zeros((len(lengths), nshuffles))

            for n in range(nshuffles):
                angle_null = np.random.rand(len(lengths)) * np.pi / 2
                fractions_n, bins = np.histogram(angle_null, thresholds)
                fractions_null[:, n] = fractions_n / np.sum(fractions_n)
                angles_null[:, n] = angle_null

            if plot:
                if axs is None:
                    f, axs = plt.subplots(1, 2, figsize=(6.5, 3.5))

                # first plot: scatter of decoding weights
                for ni in range(self.n_brains):
                    this_brain = self.which_brain == ni + 1
                    axs[0].scatter(data1[this_brain], data2[this_brain], alpha=0.5, color=pltcolors[ni],
                                   label='Brain %u' % (ni + 1), marker='.')
                    if metric == 'weight':
                        axs[0].set_xlabel('Decoding weights (%s)' % key1)
                        axs[0].set_ylabel('Decoding weights (%s)' % key2)
                    if metric == 'rate':
                        axs[0].set_xlabel('Absolute rate selectivity (%s)' % key1)
                        axs[0].set_ylabel('Absolute rate selectivity (%s)' % key2)

                axs[0].axhline([0], color='k')
                axs[0].axvline([0], color='k')
                [xm, xM] = axs[0].get_xlim()
                [ym, yM] = axs[0].get_ylim()
                axs[0].set_xlim([np.min([xm, ym]), np.max([xM, yM])])
                axs[0].set_ylim([np.min([xm, ym]), np.max([xM, yM])])

                # second plot: angle distribution
                axs[1].bar(thresholds[:-1], fractions, color=pltcolors[0], alpha=0.5, width=dx - 0.05)
                axs[1].errorbar(thresholds[:-1], np.nanmean(fractions_null, 1), 2 * np.nanstd(fractions_null, 1),
                                color='k', linestyle='', capsize=6, marker='_')
                axs[1].set_xlabel('Angle $\\theta$')
                axs[1].set_ylabel('Fraction of neurons')
                axs[1].set_xticks([-dx / 2, np.pi / 2 - dx / 2])
                axs[1].set_xticklabels(['0', '$\pi/2$'])
                sns.despine(ax=axs[0])
                sns.despine(ax=axs[1])

    return fractions, fractions_null


def selectivity_vs_decoding(self, training_fraction, sel_percentiles, cross_validations=5, nshuffles=10,
                            metric='weight', ax=None, plot=False):
    if metric == 'weight' or metric == 'reverse xor' or metric == 'weight specialization':
        self.decoding_weights = {key: [] for key in self._semantic_keys + ['XOR']}
        self.semantic_decode(training_fraction=0.9, cross_validations=cross_validations, nshuffles=0, xor=True)
        mean_decoding_weights = {key: np.abs(np.nanmean(self.decoding_weights[key], 0))[0] for key in
                                 self._semantic_keys + ['XOR']}

    sd = self._find_semantic_dichotomies()[0]

    # for the moment, let's assume only two variables
    key1 = self._semantic_keys[0]
    key2 = self._semantic_keys[1]

    if metric == 'weight':
        sel1 = np.abs(mean_decoding_weights[key1])
        sel2 = np.abs(mean_decoding_weights[key2])
        selxor = np.abs(mean_decoding_weights['XOR'])

    if metric == 'weight specialization':
        w1 = np.abs(mean_decoding_weights[key1])
        w2 = np.abs(mean_decoding_weights[key2])
        sel1 = np.abs(w1 - w2) / (w1 + w2)
        sel2 = sel1
        selxor = sel1

    if metric == 'reverse xor':
        sel1 = -np.abs(mean_decoding_weights['XOR'])
        sel2 = -np.abs(mean_decoding_weights['XOR'])
        selxor = -np.abs(mean_decoding_weights['XOR'])

    if metric == 'rate selectivity':
        rate_i_1 = np.nanmean([self.centroids['10'], self.centroids['11']], 0)
        rate_i_2 = np.nanmean([self.centroids['00'], self.centroids['01']], 0)
        sel1 = np.abs(rate_i_1 - rate_i_2)

        rate_j_1 = np.nanmean([self.centroids['01'], self.centroids['11']], 0)
        rate_j_2 = np.nanmean([self.centroids['00'], self.centroids['10']], 0)
        sel2 = np.abs(rate_j_1 - rate_j_2)

        rate_trial1 = np.nanmean([self.centroids['00'], self.centroids['11']], 0)
        rate_trial2 = np.nanmean([self.centroids['01'], self.centroids['10']], 0)

        selxor = np.abs(rate_trial1 - rate_trial2)

    if metric == 'rate specialization':
        rate_i_1 = np.nanmean([self.centroids['10'], self.centroids['11']], 0)
        rate_i_2 = np.nanmean([self.centroids['00'], self.centroids['01']], 0)
        w1 = np.abs(rate_i_1 - rate_i_2)

        rate_j_1 = np.nanmean([self.centroids['01'], self.centroids['11']], 0)
        rate_j_2 = np.nanmean([self.centroids['00'], self.centroids['10']], 0)
        w2 = np.abs(rate_j_1 - rate_j_2)

        sel1 = np.abs(w1 - w2) / (w1 + w2)
        sel2 = sel1
        selxor = sel1

    if metric == 'rate':
        rate_i_1 = np.nanmean([self.centroids['10'], self.centroids['11']], 0)
        rate_i_2 = np.nanmean([self.centroids['00'], self.centroids['01']], 0)
        rate_j_1 = np.nanmean([self.centroids['01'], self.centroids['11']], 0)
        rate_j_2 = np.nanmean([self.centroids['00'], self.centroids['10']], 0)

        sel1 = rate_i_1 + rate_i_2 + rate_j_1 + rate_j_2
        sel2 = sel1
        selxor = sel1

    data = {
        key1: [],
        key2: [],
        'XOR': []
    }

    null = {
        key1: np.zeros((len(sel_percentiles), nshuffles)),
        key2: np.zeros((len(sel_percentiles), nshuffles)),
        'XOR': np.zeros((len(sel_percentiles), nshuffles))
    }

    for i, sel_perc in tqdm(enumerate(sel_percentiles)):

        # first variable
        non_selective_cells = sel1 >= np.percentile(sel1, 100 - sel_perc)
        n_cells = np.sum(non_selective_cells)

        # decoding by using only non-selective cells - key1
        self.subset = np.where(non_selective_cells)[0]
        res = self.decode_dichotomy(sd[0], training_fraction=training_fraction,
                                    cross_validations=cross_validations)
        self._reset_random_subset()

        data[key1].append(np.nanmean(res))

        # decoding by random subsampling cells - key1
        for n in range(nshuffles):
            self._generate_random_subset(n_cells)
            res = self.decode_dichotomy(sd[0], training_fraction=training_fraction,
                                        cross_validations=cross_validations)
            self._reset_random_subset()
            null[key1][i, n] = np.nanmean(res)

        # second variable
        non_selective_cells = sel2 >= np.percentile(sel2, 100 - sel_perc)
        n_cells = np.sum(non_selective_cells)

        # decoding by using only non-selective cells - key2
        self.subset = np.where(non_selective_cells)[0]
        res = self.decode_dichotomy(sd[1], training_fraction=training_fraction,
                                    cross_validations=cross_validations)
        self._reset_random_subset()

        data[key2].append(np.nanmean(res))

        # decoding by random subsampling cells - key1
        for n in range(nshuffles):
            self._generate_random_subset(n_cells)
            res = self.decode_dichotomy(sd[1], training_fraction=training_fraction,
                                        cross_validations=cross_validations)
            self._reset_random_subset()
            null[key2][i, n] = np.nanmean(res)

        # XOR
        selective_cells = selxor >= np.percentile(selxor, 100 - sel_perc)
        n_cells = np.sum(selective_cells)

        # decoding by using only non-selective cells - key2
        self.subset = np.where(selective_cells)[0]
        xor = [['11', '00'], ['10', '01']]
        res = self.decode_dichotomy(xor, training_fraction=training_fraction,
                                    cross_validations=cross_validations)
        self._reset_random_subset()

        data['XOR'].append(np.nanmean(res))

        # decoding by random subsampling cells - key1
        for n in range(nshuffles):
            self._generate_random_subset(n_cells)
            res = self.decode_dichotomy(xor, training_fraction=training_fraction,
                                        cross_validations=cross_validations)
            self._reset_random_subset()
            null['XOR'][i, n] = np.nanmean(res)

    if plot:
        if ax is None:
            f, ax = plt.subplots(figsize=(5, 4))
        sns.despine(ax=ax)
        ax.axhline([0.5], color='k', linestyle='--')

        line_with_shade(sel_percentiles, null[key1], color=pltcolors[0], linestyle='--', axis=1, ax=ax)
        ax.plot(sel_percentiles, data[key1], color=pltcolors[0], linewidth=3, label=key1)

        line_with_shade(sel_percentiles, null[key2], color=pltcolors[1], linestyle='--', axis=1, ax=ax)
        ax.plot(sel_percentiles, data[key2], color=pltcolors[1], linewidth=3, label=key2)

        line_with_shade(sel_percentiles, null['XOR'], color=pltcolors[2], linestyle='--', axis=1, ax=ax)
        ax.plot(sel_percentiles, data['XOR'], color=pltcolors[2], linewidth=3, label='XOR')

        ax.legend()

        ax.set_xlabel('%% neurons included (from top %s)' % metric)
        ax.set_ylabel('Decoding performance')

    return data, null


def selectivity_vs_geometry(self, sel_percentiles, cross_validations=5, nshuffles=10,
                            metric='weight', ax=None, plot=False):
    if metric == 'weight' or metric == 'reverse xor' or metric == 'weight specialization':
        self.decoding_weights = {key: [] for key in self._semantic_keys + ['XOR']}
        self.semantic_decode(training_fraction=0.9, cross_validations=cross_validations, nshuffles=0, xor=True)
        mean_decoding_weights = {key: np.abs(np.nanmean(self.decoding_weights[key], 0))[0] for key in
                                 self._semantic_keys + ['XOR']}

    sd = self._find_semantic_dichotomies()[0]

    # for the moment, let's assume only two variables
    key1 = self._semantic_keys[0]
    key2 = self._semantic_keys[1]

    if metric == 'weight':
        sel1 = np.abs(mean_decoding_weights[key1])
        sel2 = np.abs(mean_decoding_weights[key2])

    if metric == 'weight specialization':
        w1 = np.abs(mean_decoding_weights[key1])
        w2 = np.abs(mean_decoding_weights[key2])
        sel1 = np.abs(w1 - w2)  # / (w1 + w2)
        sel2 = sel1

    if metric == 'reverse xor':
        sel1 = -np.abs(mean_decoding_weights['XOR'])
        sel2 = -np.abs(mean_decoding_weights['XOR'])

    if metric == 'rate selectivity':
        rate_i_1 = np.nanmean([self.centroids['10'], self.centroids['11']], 0)
        rate_i_2 = np.nanmean([self.centroids['00'], self.centroids['01']], 0)
        sel1 = np.abs(rate_i_1 - rate_i_2)

        rate_j_1 = np.nanmean([self.centroids['01'], self.centroids['11']], 0)
        rate_j_2 = np.nanmean([self.centroids['00'], self.centroids['10']], 0)
        sel2 = np.abs(rate_j_1 - rate_j_2)

    if metric == 'rate specialization':
        rate_i_1 = np.nanmean([self.centroids['10'], self.centroids['11']], 0)
        rate_i_2 = np.nanmean([self.centroids['00'], self.centroids['01']], 0)
        w1 = np.abs(rate_i_1 - rate_i_2)

        rate_j_1 = np.nanmean([self.centroids['01'], self.centroids['11']], 0)
        rate_j_2 = np.nanmean([self.centroids['00'], self.centroids['10']], 0)
        w2 = np.abs(rate_j_1 - rate_j_2)

        sel1 = np.abs(w1 - w2)  # / (w1 + w2)
        sel2 = sel1

    if metric == 'rate':
        rate_i_1 = np.nanmean([self.centroids['10'], self.centroids['11']], 0)
        rate_i_2 = np.nanmean([self.centroids['00'], self.centroids['01']], 0)
        rate_j_1 = np.nanmean([self.centroids['01'], self.centroids['11']], 0)
        rate_j_2 = np.nanmean([self.centroids['00'], self.centroids['10']], 0)

        sel1 = rate_i_1 + rate_i_2 + rate_j_1 + rate_j_2
        sel2 = sel1

    data = {
        key1: [],
        key2: [],
        'XOR': []
    }

    null = {
        key1: np.zeros((len(sel_percentiles), nshuffles)),
        key2: np.zeros((len(sel_percentiles), nshuffles)),
        'XOR': np.zeros((len(sel_percentiles), nshuffles))
    }

    for i, sel_perc in tqdm(enumerate(sel_percentiles)):

        # first variable
        selective_cells = sel1 >= np.percentile(sel1, 100 - sel_perc)
        n_cells = np.sum(selective_cells)

        # decoding by using only non-selective cells - key1
        self.subset = np.where(selective_cells)[0]
        res = self.CCGP_dichotomy(sd[0], resamplings=5)
        self._reset_random_subset()

        data[key1].append(np.nanmean(res))

        # decoding by random subsampling cells - key1
        for n in range(nshuffles):
            self._generate_random_subset(n_cells)
            res = self.CCGP_dichotomy(sd[0], resamplings=5)

            self._reset_random_subset()
            null[key1][i, n] = np.nanmean(res)

        # second variable
        selective_cells = sel2 >= np.percentile(sel2, 100 - sel_perc)
        n_cells = np.sum(selective_cells)

        # decoding by using only non-selective cells - key2
        self.subset = np.where(selective_cells)[0]
        res = self.CCGP_dichotomy(sd[1], resamplings=5)
        self._reset_random_subset()

        data[key2].append(np.nanmean(res))

        # decoding by random subsampling cells - key1
        for n in range(nshuffles):
            self._generate_random_subset(n_cells)
            res = self.CCGP_dichotomy(sd[1], resamplings=5)
            self._reset_random_subset()
            null[key2][i, n] = np.nanmean(res)

    if plot:
        if ax is None:
            f, ax = plt.subplots(figsize=(5, 4))
        sns.despine(ax=ax)
        ax.axhline([0.5], color='k', linestyle='--')

        line_with_shade(sel_percentiles, null[key1], color=pltcolors[0], linestyle='--', axis=1, ax=ax,
                        label='Random subset')
        ax.plot(sel_percentiles, data[key1], color=pltcolors[0], linewidth=3, label=key1)

        line_with_shade(sel_percentiles, null[key2], color=pltcolors[1], linestyle='--', axis=1, ax=ax,
                        label='Random subset')
        ax.plot(sel_percentiles, data[key2], color=pltcolors[1], linewidth=3, label=key2)

        ax.legend()

        ax.set_xlabel('%% neurons included (from top %s)' % metric)
        ax.set_ylabel('CCGP')

    return data, null


def decoding_correlate(self, training_fraction, block_center, block_size=50, cross_validations=5, metric='weight'):
    if metric == 'weight' or metric == 'specialization':
        for key in self._semantic_keys:
            self.decoding_weights[key] = []

        self.semantic_decode(training_fraction=0.9, cross_validations=cross_validations, nshuffles=0)

        mean_decoding_weights = {key: np.abs(np.nanmean(self.decoding_weights[key], 0))[0] for key in
                                 self._semantic_keys}

    sd = self._find_semantic_dichotomies()[0]

    # for the moment, let's assume only two variables
    key1 = self._semantic_keys[0]
    key2 = self._semantic_keys[1]

    if metric == 'weight':
        sel1 = mean_decoding_weights[key1] / np.sqrt(np.sum(mean_decoding_weights[key1] ** 2.))
        sel2 = mean_decoding_weights[key2] / np.sqrt(np.sum(mean_decoding_weights[key2] ** 2.))

    if metric == 'rate':
        rate_i_1 = np.nanmean([self.centroids['10'], self.centroids['11']], 0)
        rate_i_2 = np.nanmean([self.centroids['00'], self.centroids['01']], 0)
        sel1 = np.abs(rate_i_1 - rate_i_2)

        rate_j_1 = np.nanmean([self.centroids['01'], self.centroids['11']], 0)
        rate_j_2 = np.nanmean([self.centroids['00'], self.centroids['10']], 0)
        sel2 = np.abs(rate_j_1 - rate_j_2)

    if metric == 'specialization':
        sel1 = mean_decoding_weights[key1] / np.sqrt(np.sum(mean_decoding_weights[key1] ** 2.))
        sel2 = mean_decoding_weights[key2] / np.sqrt(np.sum(mean_decoding_weights[key2] ** 2.))
        spec = sel1 - sel2
        sel1 = spec
        sel2 = -spec

    if metric == 'random':
        sel1 = np.random.rand(self.n_neurons)
        sel2 = np.random.rand(self.n_neurons)

    data = {}

    # first variable
    selected_cells = (np.percentile(sel1, block_center - 0.5 * block_size) < sel1) & (
            sel1 <= np.percentile(sel1, block_center + 0.5 * block_size))
    self.subset = np.where(selected_cells)[0]
    res = self.decode_dichotomy(sd[0], training_fraction=training_fraction,
                                cross_validations=cross_validations)
    self._reset_random_subset()
    data[key1] = np.nanmean(res)

    # second variable
    selected_cells = (np.percentile(sel2, block_center - 0.5 * block_size) < sel2) & (
            sel2 <= np.percentile(sel2, block_center + 0.5 * block_size))
    self.subset = np.where(selected_cells)[0]
    res = self.decode_dichotomy(sd[1], training_fraction=training_fraction,
                                cross_validations=cross_validations)
    self._reset_random_subset()
    data[key2] = np.nanmean(res)

    return data


# Single neurons features analysis


def decode_with_subsamples(dec, training_fraction, n_neurons, nreps, cross_validations=5, ndata='auto', plot=False,
                           ax=None, xor=False):
    results = {}
    n_neurons = np.copy(n_neurons)
    n_neurons = n_neurons[n_neurons <= dec.n_neurons]

    for key in dec._semantic_keys:
        results[key] = []
    if xor:
        results['XOR'] = []

    nflag = False
    if ndata == 'auto':
        nflag = True

    for n in n_neurons:
        if nflag:
            ndata = 2 * n

        print(ndata)

        results_n = {}
        for key in dec._semantic_keys:
            results_n[key] = []
        if xor:
            results_n['XOR'] = []

        for i in range(nreps):
            dec._generate_random_subset(n)

            perfs, _ = dec.semantic_decode(training_fraction, cross_validations, nshuffles=0, ndata=ndata, xor=xor)

            for key in perfs.keys():
                results_n[key].append(np.nanmean(perfs[key]))
            dec._reset_random_subset()

        for key in perfs.keys():
            results[key].append(results_n[key])

    if plot:
        if ax is None:
            f, ax = plt.subplots(figsize=(6, 5))
            ax.set_ylabel('Decoding Performance')
            ax.set_xlabel('N neurons')
            sns.despine(ax=ax)
            ax.axhline([0.5], color='k', linestyle='--')
        for key in dec._semantic_keys:
            ax.errorbar(n_neurons, np.nanmean(results[key], 1), np.nanstd(results[key], 1),
                        label=key, marker='o', capsize=5, alpha=0.6)
        plt.legend()

    return n_neurons, results


# Geometrical features

def compute_parallelism_score(dec, nshuffles=10, plot=False, ax=None, xor=False):
    semantic_dics, semantic_keys = dec._find_semantic_dichotomies()

    if xor and len(dec.conditions) == 2:
        semantic_dics.append([['01', '10'], ['00', '11']])
        semantic_keys.append('XOR')

    parallelism = {}
    parallelism_null = {key: [] for key in semantic_keys}

    for key, dic in zip(semantic_keys, semantic_dics):
        selectivity_vectors = []

        for w_top in dic[0]:
            for w_bottom in dic[1]:
                if hamming(string_bool(w_top), string_bool(w_bottom)) == 1:
                    selectivity_vectors.append(dec.centroids[w_top] - dec.centroids[w_bottom])
        cosines = []

        for i in range(len(selectivity_vectors)):
            for j in range(i + 1, len(selectivity_vectors)):
                cosines.append(cosine(selectivity_vectors[i], selectivity_vectors[j]))
        parallelism[key] = np.nanmean(cosines)

        if dec._verbose and nshuffles:
            count = tqdm(range(nshuffles))
        else:
            count = range(nshuffles)

        for n in count:
            dec._rototraslate_conditioned_rasters()
            dec._compute_centroids()

            selectivity_vectors = []
            for w_top in dic[0]:
                for w_bottom in dic[1]:
                    if hamming(string_bool(w_top), string_bool(w_bottom)) == 1:
                        selectivity_vectors.append(dec.centroids[w_top] - dec.centroids[w_bottom])
            cosines = []
            for i in range(len(selectivity_vectors)):
                for j in range(i + 1, len(selectivity_vectors)):
                    cosines.append(cosine(selectivity_vectors[i], selectivity_vectors[j]))

            parallelism_null[key].append(np.nanmean(cosines))
            dec._order_conditioned_rasters()
            dec._compute_centroids()

    # if xor and len(self.semantic_conditions) == 2:
    #     parallelism_null['XOR'] = [np.nan]

    if plot:
        if not ax:
            f, ax = plt.subplots(figsize=(2 * len(semantic_dics), 4))
        plot_perfs_null_model(parallelism, parallelism_null, marker='$//$', ylabel='Parallelism Score', ax=ax,
                              shownull='violin', chance=0)

    return parallelism, parallelism_null


def compute_square_score(dec, nshuffles=10, plot=False, axs=None):
    deltas_sem = []
    deltas_neur = []

    sem_distances = cdist(dec._condition_vectors, dec._condition_vectors)
    neur_distances = mahalanobis_dissimilarity(dec)

    for i in range(len(dec._condition_vectors)):
        for j in range(i + 1, len(dec._condition_vectors)):
            deltas_sem.append(sem_distances[i, j])
            deltas_neur.append(neur_distances[i, j])

    slope, intercept, r, p, se = linregress(deltas_sem, deltas_neur)
    r_data = r

    if plot:
        if axs is None:
            f, axs = plt.subplots(1, 2, figsize=(7, 4), gridspec_kw={'width_ratios': [2.5, 1]})
            sns.despine(f)
        corr_scatter(deltas_sem, deltas_neur, '$\Delta r_{semantic}$', '$\Delta r_{neural}$', ax=axs[0], alpha=0.4)

    r_null = []

    if dec._verbose and nshuffles:
        count = tqdm(range(nshuffles))
    else:
        count = range(nshuffles)

    for n in count:
        dec._rototraslate_conditioned_rasters()
        dec._compute_centroids()
        deltas_sem = []
        deltas_neur = []
        neur_distances = mahalanobis_dissimilarity(dec)

        for i in range(len(dec._condition_vectors)):
            for j in range(i + 1, len(dec._condition_vectors)):
                deltas_sem.append(sem_distances[i, j])
                deltas_neur.append(neur_distances[i, j])

        slope, intercept, r, p, se = linregress(deltas_sem, deltas_neur)
        r_null.append(r)
        dec._order_conditioned_rasters()
        dec._compute_centroids()

    if plot:
        plot_perfs_null_model({'Semantic corr': r_data}, {'Semantic corr': r_null}, marker='s', ylabel='',
                              ax=axs[1], shownull='violin', chance=0, setup=False)

    return r_data, r_null


def compute_planarity(dec, nshuffles=10, plot=False, ax=None):
    centroids = list(dec.centroids.values())
    planar_distances = interplanar_distance(centroids)
    data = 1 - np.nanmean(planar_distances)

    shuffled = []
    if dec._verbose and nshuffles:
        count = tqdm(range(nshuffles))
    else:
        count = range(nshuffles)

    for n in count:
        dec._rototraslate_conditioned_rasters()
        dec._compute_centroids()
        centroids = list(dec.centroids.values())
        planar_distances = interplanar_distance(centroids)
        shuffled.append(1 - np.nanmean(planar_distances))  # np.exp(-np.nanmean(planar_distances)))

        dec._order_conditioned_rasters()
        dec._compute_centroids()

    if plot:
        if ax is None:
            f, ax = plt.subplots(figsize=(3, 4))
            setup_decoding_axis(ax, 'Planarity', ylow=-0.1, null=0, yhigh=0.6)
        plot_perfs_null_model({'Planarity': data}, {'Planarity': shuffled}, marker='^', ylabel='', ax=ax,
                              shownull='violin', chance=0, setup=False)
    return data, shuffled


def geometry_analysis(dec, training_fraction=0.8,
                      nshuffles=10,
                      nshuffles_noncv=101,
                      cross_validations=10,
                      parallel=False,
                      plot=False,
                      data=None,
                      null=None,
                      names=None,
                      savename=None,
                      title=''):
    if data is None and null is None:
        data = {}
        null = {}

    if plot:
        mpl.rcParams.update({'figure.autolayout': False})
        fig = plt.figure(figsize=(12, 10))
        G = GridSpec(20, 24)

        ax3d = fig.add_subplot(G[:12, 0:12], projection='3d')
        if len(dec.conditions) == 2:
            ax_decoding = fig.add_subplot(G[2:9, 12:18])
            ax_ccgp = fig.add_subplot(G[2:9, 20:24])

            ax_scatter = fig.add_subplot(G[12:17, 2:8])
            ax_square = fig.add_subplot(G[11:17, 10:13])
            ax_plan = fig.add_subplot(G[11:17, 15:18])
            ax_par = fig.add_subplot(G[11:17, 20:24])

        if len(dec.conditions) == 3:
            ax_decoding = fig.add_subplot(G[2:9, 12:17])
            ax_ccgp = fig.add_subplot(G[2:9, 19:24])

            ax_scatter = fig.add_subplot(G[12:17, 2:8])
            ax_square = fig.add_subplot(G[11:17, 10:13])
            ax_plan = fig.add_subplot(G[11:17, 14:17])
            ax_par = fig.add_subplot(G[11:17, 19:24])

        ax_decoding.set_title(title)
        sns.despine(fig)

        # non-CV quantities
        if dec._verbose:
            print("[Geometry analysis]\t computing Squarity")
        data['Squore'], null['Squore'] = dec.compute_square_score(nshuffles=nshuffles_noncv, plot=True,
                                                                  axs=[ax_scatter, ax_square])

        if dec._verbose:
            print("[Geometry analysis]\t computing Parallelism Score")
        data['Parallel'], null['Parallel'] = dec.compute_parallelism_score(nshuffles=nshuffles_noncv, plot=True,
                                                                           ax=ax_par)

        if dec._verbose:
            print("[Geometry analysis]\t computing Planarity")
        data['Planarity'], null['Planarity'] = dec.compute_planarity(nshuffles=nshuffles_noncv, plot=True,
                                                                     ax=ax_plan)

        # CV quantities
        if 'Decoding' not in data.keys():
            if dec._verbose:
                print("[Geometry analysis]\t computing Decoding Performance")
            data['Decoding'], null['Decoding'] = dec.semantic_decode(training_fraction=training_fraction,
                                                                     cross_validations=cross_validations,
                                                                     nshuffles=nshuffles, xor=True,
                                                                     parallel=parallel)
        if 'CCGP' not in data.keys():
            if dec._verbose:
                print("[Geometry analysis]\t computing CCGP")
            data['CCGP'], null['CCGP'] = dec.semantic_CCGP(nshuffles=nshuffles, ntrials=3)

        if savename is not None:
            dec.visualize_decodanda_MDS(data=data, null=null, names=names, axs=[ax3d, ax_decoding, ax_ccgp],
                                        savename=savename + '.gif')
            fig.savefig(savename + '.pdf')
        else:
            dec.visualize_decodanda_MDS(data=data, null=null, names=names, axs=[ax3d, ax_decoding, ax_ccgp])

        mpl.rcParams.update({'figure.autolayout': True})


    else:
        # non-CV quantities
        if 'Squore' not in data.keys():
            data['Squore'], null['Squore'] = dec.compute_square_score(nshuffles=nshuffles_noncv, plot=False)

        if 'Parallel' not in data.keys():
            data['Parallel'], null['Parallel'] = dec.compute_parallelism_score(nshuffles=nshuffles_noncv,
                                                                               plot=False)

        if 'Planarity' not in data.keys():
            data['Planarity'], null['Planarity'] = dec.compute_planarity(nshuffles=nshuffles_noncv, plot=False)

        # CV quantities
        if 'Decoding' not in data.keys():
            data['Decoding'], null['Decoding'] = dec.semantic_decode(training_fraction=training_fraction,
                                                                     cross_validations=cross_validations,
                                                                     nshuffles=nshuffles, xor=True)
        if 'CCGP' not in data.keys():
            data['CCGP'], null['CCGP'] = dec.semantic_CCGP(nshuffles=nshuffles, ntrials=3)

    return data, null

# This is CCGP with the old null model
def CCGP_dichotomy_old(self, dichotomy, ntrials=3, ndata='auto', only_semantic=True, shuffled=False,
                       destroy_correlations=False):
    """

    Parameters
    ----------
    dichotomy
    ntrials
    ndata
    only_semantic
    shuffled
    destroy_correlations

    Returns
    -------

    """
    # TODO: make these comments into proper doc
    # dic is in the form of a 2xL list, where L is the number of condition vectors in a dichtomy
    # Example: dic = [['10', '11'], ['00', '01']]
    #
    # CCGP analysis works by choosing one condition vector from each class of the dichotomies, train over
    # the remaining L-1 vs L-1, and use the two selected condition vectors for testing
    if type(dichotomy) == str:
        dic = self._dichotomy_from_key(dichotomy)
    else:
        dic = dichotomy

    if ndata == 'auto' and self.n_brains == 1:
        ndata = self._max_conditioned_data
    if ndata == 'auto' and self.n_brains > 1:
        ndata = max(self._max_conditioned_data, 2 * self.n_neurons)

    if self._verbose and not shuffled:
        log_dichotomy(self, dic, ndata, 'Cross-condition decoding')

    if shuffled:
        self._rototraslate_conditioned_rasters()
    else:
        self._print('\nLooping over CCGP sampling repetitions:')

    all_performances = []
    if not shuffled and self._verbose:
        iterable = tqdm(range(ntrials))
    else:
        iterable = range(ntrials)

    for n in iterable:
        performances = []

        set_A = dic[0]
        set_B = dic[1]

        for i in range(len(set_A)):
            for j in range(len(set_B)):
                test_condition_A = set_A[i]
                test_condition_B = set_B[j]
                # TODO: do we need the only_semantic keyword?
                if only_semantic:
                    go = (hamming(string_bool(test_condition_A), string_bool(test_condition_B)) == 1)
                else:
                    go = True
                if go:
                    training_conditions_A = [x for iA, x in enumerate(set_A) if iA != i]
                    training_conditions_B = [x for iB, x in enumerate(set_B) if iB != j]

                    training_array_A = []
                    training_array_B = []
                    label_A = ''
                    label_B = ''

                    for ck in training_conditions_A:
                        arr = sample_from_rasters(self.conditioned_rasters[ck], ndata=ndata)
                        training_array_A.append(arr)
                        label_A += (self._semantic_vectors[ck] + ' ')

                    for ck in training_conditions_B:
                        arr = sample_from_rasters(self.conditioned_rasters[ck], ndata=ndata)
                        training_array_B.append(arr)
                        label_B += (self._semantic_vectors[ck] + ' ')

                    training_array_A = np.vstack(training_array_A)
                    training_array_B = np.vstack(training_array_B)

                    testing_array_A = sample_from_rasters(self.conditioned_rasters[test_condition_A], ndata=ndata)
                    testing_array_B = sample_from_rasters(self.conditioned_rasters[test_condition_B], ndata=ndata)

                    if destroy_correlations:
                        destroy_time_correlations(training_array_A)
                        destroy_time_correlations(training_array_B)
                        destroy_time_correlations(testing_array_A)
                        destroy_time_correlations(testing_array_B)

                    self._train(training_array_A, training_array_B, label_A, label_B)
                    performance = self._test(testing_array_A, testing_array_B, label_A, label_B)
                    performances.append(performance)

        all_performances.append(np.nanmean(performances))
    if shuffled:
        self._order_conditioned_rasters()
    return all_performances

# Visualization


def plot_perfs(perfs_in, labels=None, x=0, ax=None, color=None, marker='o', alpha=0.8, s=50, errorbar=True,
               annotate=True,
               null=0.5, labelfontsize=9, linepadding=None, ptype='t', null_data=None):
    perfs = np.asarray(perfs_in)
    if null_data is not None:
        null_means = np.nanmean(null_data, 1)

    if not ax:
        f, ax = plt.subplots()
    if not color:
        ax.scatter(np.ones(len(perfs)) * x, perfs, s=s, alpha=alpha, marker=marker)
    else:
        ax.scatter(np.ones(len(perfs)) * x, perfs, s=s, alpha=alpha, marker=marker, facecolor='w', edgecolors=color)

    if errorbar:
        ax.errorbar([x], np.nanmean(perfs), 2 * np.nanstd(perfs), color='k', linewidth=1, capsize=6,
                    marker='_', alpha=0.5)

    if annotate:
        nonnan = np.isnan(perfs) == 0
        if ptype == 't':
            t, pval = ttest_1samp(perfs[nonnan], null)
        if ptype == 'paired_w':
            t, pval = wilcoxon(perfs[nonnan], null_means[nonnan])
        if ptype == 'paired_t':
            t, pval = ttest_rel(perfs[nonnan], null_means[nonnan])
        if ptype == 'ttest_z':
            zs = [(perfs[i] - np.nanmean(null_data[i])) / np.nanstd(null_data[i]) for i in range(len(perfs))]
            t, pval = ttest_1samp(zs, 0)
        if ptype == 'multi_z':
            zs = [(perfs[i] - np.nanmean(null_data[i])) / np.nanstd(null_data[i]) for i in range(len(perfs))]
            pval = scipy.stats.norm.sf(np.abs(np.nanmean(zs)))
            t = np.nan

        if linepadding is None:
            if np.nanmean(perfs) > null:
                linepadding = 0.01
            else:
                linepadding = -0.01

        ax.plot([x - 0.15, x - 0.15, x - 0.12], [null + linepadding, np.nanmean(perfs), np.nanmean(perfs)], color='k',
                alpha=0.5, linewidth=0.5)
        ax.text(x - 0.14, 0.5 * (np.nanmean(perfs) + null), p_to_ast(pval), rotation=90, ha='right', va='center',
                fontsize=14)

    if labels is not None:
        for i in range(len(perfs)):
            ax.text(x + 0.05, perfs[i], labels[i], fontsize=labelfontsize)



def corr_scat_kde(x, y, ax=None, xlabel=None, ylabel=None, **kwargs):
    ax = ax or plt.gca()
    sns.scatterplot(x, y, ax=ax, **kwargs)
    sns.kdeplot(x, y, ax=ax, kind='kde', alpha=0.7)
    corrfunc(x, y, ax=ax)
    sns.despine(ax=ax)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax

