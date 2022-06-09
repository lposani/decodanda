import numpy as np

from .imports import *
from .utilities import *


def visualize_raster(raster, ax='auto', offset=0, order=None, colors=None):
    if order is None:
        order = np.arange(raster.shape[1], dtype=int)
    if colors is None:
        colors = np.zeros(raster.shape[1], dtype=int)
    if ax == 'auto':
        f, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlabel('Time bin')
        ax.set_ylabel('Neuron index')
        ax.set_ylim([0, raster.shape[1]])
    for i in range(raster.shape[1]):
        d = np.where(raster[:, order[i]] > 0)[0]
        ax.plot(d+offset, np.ones(len(d)) * i, linestyle='', marker='|', markersize=200.0 / raster.shape[1],
                alpha=0.8, color=pltcolors[colors[order[i]]-1])
    return ax


def setup_decoding_axis(ax, labels, ylow=0.4, yhigh=1.0, null=0.5):
    ax.set_ylabel('Decoding performance')
    ax.axhline([null], linestyle='--', color='k')
    ax.set_xticks(range(len(labels)))
    if len(labels):
        ax.set_xlim([-0.5, len(labels)-0.5])
    ax.set_xticklabels(labels, rotation=45, ha='right')

    for i in range(1, int((yhigh-null)/0.1)+1):
        ax.axhline([null + i*0.1], linestyle='-', color='k', alpha=0.1)
        ax.axhline([null - i * 0.1], linestyle='-', color='k', alpha=0.1)
    ax.set_ylim([ylow, yhigh + 0.04])
    sns.despine(ax=ax)


def plot_perfs(perfs_in, labels=None, x=0, ax=None, color=None, marker='o', alpha=0.8, s=50, errorbar=True, annotate=True,
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
            zs = [(perfs[i] - np.nanmean(null_data[i]))/np.nanstd(null_data[i]) for i in range(len(perfs))]
            t, pval = ttest_1samp(zs, 0)
        if ptype == 'multi_z':
            zs = [(perfs[i] - np.nanmean(null_data[i]))/np.nanstd(null_data[i]) for i in range(len(perfs))]
            pval = scipy.stats.norm.sf(np.abs(np.nanmean(zs)))
            t = np.nan

        if linepadding is None:
            if np.nanmean(perfs) > null:
                linepadding = 0.01
            else:
                linepadding = -0.01

        ax.plot([x - 0.15, x - 0.15, x - 0.12], [null+linepadding, np.nanmean(perfs), np.nanmean(perfs)], color='k',
                alpha=0.5, linewidth=0.5)
        ax.text(x - 0.14, 0.5 * (np.nanmean(perfs) + null), p_to_ast(pval), rotation=90, ha='right', va='center',
                fontsize=14)

    if labels is not None:
        for i in range(len(perfs)):
            ax.text(x+0.05, perfs[i], labels[i], fontsize=labelfontsize)


def plot_perfs_null_model(perfs, perfs_nullmodel, marker='d', ylabel='Decoding performance', ax=None, shownull=False, chance=0.5, setup=True, ptype='count'):
    labels = list(perfs.keys())
    pvals = {}
    if not ax:
        f, ax = plt.subplots(figsize=(2 * len(perfs), 4))
    if shownull == 'swarm':
        sns.swarmplot(ax=ax, data=pd.DataFrame(perfs_nullmodel, columns=labels), alpha=0.2, size=4, color='k')
    if shownull == 'violin':
        sns.violinplot(ax=ax, data=pd.DataFrame(perfs_nullmodel, columns=labels), color=[0.8, 0.8, 0.8, 0.3], bw=.25, cut=0, inner=None)
    if setup:
        setup_decoding_axis(ax, labels, ylow=0.28, yhigh=1.02, null=chance)
    ax.set_ylabel(ylabel)

    for i, l in enumerate(labels):
        if ptype == 'count':
            top_quartile = np.percentile(perfs_nullmodel[l], 95) - np.nanmean(perfs_nullmodel[l])
            low_quartile = np.nanmean(perfs_nullmodel[l]) - np.percentile(perfs_nullmodel[l], 5)
            ax.errorbar([i], np.nanmean(perfs_nullmodel[l]), yerr=np.asarray([[low_quartile, top_quartile]]).T, color='k',
                        linewidth=2, capsize=5, marker='_', alpha=0.3)
            pval = np.nanmean(np.asarray(perfs_nullmodel[l]) > perfs[l])

        if ptype == 'zscore' or ptype == 'z':
            ax.errorbar([i], np.nanmean(perfs_nullmodel[l]), yerr=2*np.nanstd(perfs_nullmodel[l]), color='k',
                        linewidth=2, capsize=5, marker='_', alpha=1.0)
            pval = z_pval(perfs[l], perfs_nullmodel[l])[1]

        ax.scatter([i], [perfs[l]], marker=marker, s=100, color=pltcolors[i], facecolor='none', linewidth=2)

        pvals[l] = pval
        if pval < 0.05:
            ax.plot([i - 0.2, i - 0.22, i - 0.22, i - 0.2],
                    [np.nanmean(perfs_nullmodel[l]), np.nanmean(perfs_nullmodel[l]), perfs[l], perfs[l]],
                    color='k', alpha=0.5, linewidth=1.0)
            ax.text(i - 0.18, 0.5 * (np.nanmean(perfs_nullmodel[l]) + perfs[l]), p_to_ast(pval), rotation=90,
                    ha='right', va='center', fontsize=14)
    return pvals


def plot_perfs_null_model_single(data, null, x=0, marker='d', ax=None, shownull=False, color='b', ptype='zscore'):
    if not ax:
        f, ax = plt.subplots(figsize=(3, 4))
        sns.despine(ax=ax)

    if shownull == 'box':
        plt.boxplot(x=null, positions=[x], showfliers=False, notch=True, manage_ticks=False)

    if ptype == 'count':
        top_quartile = np.percentile(null, 95) - np.nanmean(null)
        low_quartile = np.nanmean(null) - np.percentile(null, 5)
        ax.errorbar([x], np.nanmean(null), yerr=np.asarray([[low_quartile, top_quartile]]).T, color='k',
                    linewidth=2, capsize=5, marker='_', alpha=0.3)
        pval = np.nanmean(np.asarray(null) > data)

    if ptype == 'zscore':
        ax.errorbar([x], np.nanmean(null), yerr=2 * np.nanstd(null), color='k',
                    linewidth=2, capsize=5, marker='_', alpha=0.3)
        z, pval = z_pval(data, null)
        print('Z = ', z, 'P = ', pval)

    ax.scatter([x], [data], marker=marker, s=100, color=color, facecolor='white', linewidth=2)

    ax.plot([x - 0.2, x - 0.22, x - 0.22, x - 0.2],
            [np.nanmean(null), np.nanmean(null), data, data],
            color='k', alpha=0.5, linewidth=1.0)
    ax.text(x - 0.18, 0.5 * (np.nanmean(null) + data), p_to_ast(pval), rotation=90,
            ha='right', va='center', fontsize=14)

    return pval


def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    slope, intercept, r, p, err = scipy.stats.linregress(x, y)
    s, psp = scipy.stats.spearmanr(x, y)
    ax = ax or plt.gca()
    if p<0.05 or psp<0.05:
        ax.annotate(f'r = {r:.2f}\n{p_to_ast(p):s}\nρ = {s:.2f}\n{p_to_ast(psp):s}\nn = {len(x):.0f}', xy=(.75, .01), xycoords=ax.transAxes, fontsize=8, color='k')
    else:
        ax.annotate(f'r = {r:.2f}\n{p_to_ast(p):s}\nρ = {s:.2f}\n{p_to_ast(psp):s}\nn = {len(x):.0f}', xy=(.75, .01), xycoords=ax.transAxes, fontsize=8, color='k')

    xs = ax.get_xlim()
    ax.plot(xs, slope * np.asarray(xs) + intercept, color=pltcolors[1], linewidth=2, alpha=0.5)


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


def corr_scatter(x_data, y_data, xlabel, ylabel, ax=None, data_labels=None, corr=None, annotate=True, **kwargs):
    nanmask = (np.isnan(x_data) == 0) & (np.isnan(y_data) == 0)
    x = np.asarray(x_data)[nanmask]
    y = np.asarray(y_data)[nanmask]
    if corr is None:
        corr = scipy.stats.pearsonr
    returnax = False
    if ax is None:
        f, ax = plt.subplots(figsize=(4, 4))
        returnax = True
    slope, intercept, r, p, err = scipy.stats.linregress(x, y)
    r, p = corr(x, y)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.scatter(x, y, **kwargs)
    ys = ax.get_ylim()
    xs = ax.get_xlim()
    # ax.plot(xs, slope*np.asarray(xs)+intercept, color=pltcolors[1])
    if annotate:
        corrfunc(x, y, ax=ax)
        # if p<0.05:
        #     ax.text(xs[1], slope*np.asarray(xs)[1]+intercept, '%s\nr=%.2f' % (p_to_text(p), r), va='center', ha='center', color='r')
        # else:
        #     ax.text(xs[1], slope*np.asarray(xs)[1]+intercept, '%s\nr=%.2f' % (p_to_text(p), r), va='center', ha='center', color='k')
    ax.set_ylim(ys)
    ax.set_xlim([xs[0], xs[1]+(xs[1]-xs[0])*0.05])
    sns.despine(ax=ax)

    if data_labels is not None:
        for i in range(len(x)):
            ax.text(x[i], y[i], ' '+data_labels[i])
    if returnax:
        return r, p, f, ax
    else:
        return r, p


def line_with_shade(x, y, errfunc=np.nanstd, ax=None, axis=0, label='', color='k', alpha=0.1, **kwargs):
    if ax is None:
        f, ax = plt.subplots(figsize=(5,4))

    mean = np.nanmean(y, axis)
    err = errfunc(y, axis)
    ax.plot(x, mean, color=color, label=label, **kwargs)
    ax.fill_between(x, mean-err, mean+err, color=color, alpha=alpha)
    ax.plot(x, mean+err, linewidth=0.5, color=color, alpha=alpha+0.2)
    ax.plot(x, mean-err, linewidth=0.5, color=color, alpha=alpha+0.2)
    return ax


def smooth_hist(data, ax, bins, stairs=False, label=None, color='k'):
    nonnan = np.isnan(data) == 0
    arraydata = np.asarray(data)[nonnan]
    density = scipy.stats.gaussian_kde(arraydata)
    delta = np.nanmax(arraydata) - np.nanmin(arraydata)
    x = np.linspace(np.nanmin(arraydata) - delta / 5., np.nanmax(arraydata) + delta / 5., bins)
    if stairs:
        n, x, _ = ax.hist(arraydata, bins=bins, histtype=u'step', density=True)
    ax.plot(x, density(x), label=label, color=color)



# Decoding visualization

def visualize_decoding(dec, dic, perfs, null, ndata=100, training_fraction=0.5, testing_trials=None):
    f = plt.figure(figsize=(18, 9))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    gs = GridSpec(25, 39, figure=f)

    mfrA = []
    mfrB = []

    # Retrieving names
    dic_key = dec._dic_key(dic)
    f.suptitle('Decoding %s' % dic_key, fontsize=16)
    set_A = dic[0]
    label_A = ''
    for d in set_A:
        label_A += (dec.semantic_vectors[d] + ' ')
        mfrA.append(dec.centroids[d])
    label_A = label_A[:-1]
    mfrA = np.nanmean(mfrA, 0)

    set_B = dic[1]
    label_B = ''
    for d in set_B:
        label_B += (dec.semantic_vectors[d] + ' ')
        mfrB.append(dec.centroids[d])
    label_B = label_B[:-1]
    mfrB = np.nanmean(mfrB, 0)

    # Plotting A and B conditions
    def plot_dic_side(dic_side, ax, lab):
        x = 0
        for key in dic_side:
            y = 0
            xnew = 0
            for brain_index in range(dec.n_brains):
                for r in dec.conditioned_rasters[key][brain_index].T:
                    y +=1
                    ax.plot(np.arange(len(r))+x, y + 3*r/ np.nanmax(dec.conditioned_rasters[key][brain_index]), color=pltcolors[brain_index], alpha=0.5)
                    xnew = max(xnew, len(r))
            ax.text(x, y*1.03, dec.semantic_vectors[key])
            x = x + xnew*1.1
        sns.despine(ax=ax)
        ax.set_ylabel('Neuron #')
        ax.set_xlabel('Time bin')
        ax.set_title(lab)
    axA = f.add_subplot(gs[:, 0:10])
    plot_dic_side(dic[0], axA, '%s = %s' % (dic_key, list(dec.conditions[dic_key].keys())[1]))
    axB = f.add_subplot(gs[:, 12:22])
    plot_dic_side(dic[1], axB,'%s = %s' % (dic_key, list(dec.conditions[dic_key].keys())[0]))

    # plotting decoding performance
    perf = np.nanmean(perfs)
    axD = f.add_subplot(gs[:7, 25:])
    kde = scipy.stats.gaussian_kde(null)
    null_x = np.linspace(0., 1.0, 100)
    null_y = kde(np.linspace(0.1, 0.9, 100))
    axD.plot(null_x, null_y, color='k', alpha=0.5)
    axD.fill_between(null_x, null_y, color='k', alpha=0.3)
    axD.text(null_x[np.argmax(null_y)], np.max(null_y)*1.05, 'null model', ha='right')
    sns.despine(ax=axD)
    axD.plot([perf, perf], [0, np.max(null_y)], color='red', linewidth=3)
    axD.text(perf, np.max(null_y)*1.05, 'data', ha='left', color='red')
    axD.set_xlabel('Decoding Performance (%s)' % dic_key)
    null_mean = np.nanmean(null)
    z, p = z_pval(perf, null)
    axD.text(1.0, 0.05*np.max(null_y), '%s\nz=%.1f\nP=%.1e' % (p_to_ast(p), z, p), ha='center')
    axD.plot(null, np.zeros(len(null)), linestyle='', marker='|', color='k')
    axD.plot(perfs, np.zeros(len(perfs)), linestyle='', marker='.', color='r', alpha=0.5)
    axD.plot([null_mean, null_mean], [0, kde(null_mean)], color='k', linestyle='--')

    # plotting decoding weight distribution
    axW = f.add_subplot(gs[9:16, 25:31])
    sns.despine(ax=axW)
    Ws = np.squeeze(np.nanmean(dec.decoding_weights[dic_key], 0).T)
    bins = np.linspace(-np.nanmax(np.abs(Ws)), np.nanmax(np.abs(Ws)), 25)
    axW.hist(Ws, bins, density=True, alpha=0.6, color='k')
    kde = scipy.stats.gaussian_kde(Ws)
    W_x = np.linspace(np.nanmin(Ws)-(np.nanmax(Ws)-np.nanmin(Ws))*0.1, np.nanmax(Ws)+(np.nanmax(Ws)-np.nanmin(Ws))*0.1, 100)
    W_y = kde(W_x)
    axW.plot(W_x, W_y, color='k', linewidth=2, alpha=0.8)
    for n in range(dec.n_brains):
        axW.hist(Ws[dec.which_brain == n+1], bins, color=pltcolors[n], density=True, alpha=0.6)
    axW.set_xlabel('Decoding Weight')
    axW.axvline([0], color='k', linestyle='--')

    # plotting selectivity distribution
    axS = f.add_subplot(gs[9:16, 33:])
    sns.despine(ax=axS)
    selectivity = (mfrA - mfrB)/(mfrA + mfrB)
    selectivity[np.isnan(selectivity)] = 0
    selectivity[np.isinf(selectivity)] = 0
    bins = np.linspace(-np.nanmax(np.abs(selectivity)), np.nanmax(np.abs(selectivity)), 25)
    kde = scipy.stats.gaussian_kde(selectivity)
    W_x = np.linspace(np.nanmin(selectivity) - (np.nanmax(selectivity) - np.nanmin(selectivity)) * 0.1,
                      np.nanmax(selectivity) + (np.nanmax(selectivity) - np.nanmin(selectivity)) * 0.1, 100)
    W_y = kde(W_x)
    axS.plot(W_x, W_y, color='k', linewidth=2, alpha=0.8)
    for n in range(dec.n_brains):
        axS.hist(selectivity[dec.which_brain == n + 1], bins, color=pltcolors[n], density=True, alpha=0.6)
    axS.set_xlabel('Neuron Selectivity')
    axS.axvline([0], color='k', linestyle='--')
    selectivity = np.abs(selectivity)

    # plotting mean firing rate
    axWS = f.add_subplot(gs[18:, 25:31])
    sns.despine(ax=axWS)
    keyA = list(dec.conditions[dic_key].keys())[1]
    keyB = list(dec.conditions[dic_key].keys())[0]

    corr_scatter(mfrA, mfrB, 'MFR (%s)' % keyA, 'MFR (%s)' % keyB, color='k', alpha=0.0, ax=axWS)
    for n in range(dec.n_brains):
        axWS.scatter(mfrA[dec.which_brain == n + 1], mfrB[dec.which_brain == n + 1], color=pltcolors[n], alpha=0.4)

    # plotting training and testing example selectivity
    training_array_A = []
    training_array_B = []
    testing_array_A = []
    testing_array_B = []

    if ndata == 'auto':
        if ndata == 'auto' and dec.n_brains == 1:
            ndata = dec.max_conditioned_data
        if ndata == 'auto' and dec.n_brains > 1:
            ndata = max(dec.max_conditioned_data, 2 * dec.n_neurons)

    for d in set_A:
        training, testing = sample_training_testing_from_rasters(dec.conditioned_rasters[d],
                                                                 ndata,
                                                                 training_fraction,
                                                                 dec.conditioned_trial_index[d],
                                                                 testing_trials=testing_trials)

        training_array_A.append(training)
        testing_array_A.append(testing)

    for d in set_B:
        training, testing = sample_training_testing_from_rasters(dec.conditioned_rasters[d],
                                                                 ndata,
                                                                 training_fraction,
                                                                 dec.conditioned_trial_index[d],
                                                                 testing_trials=testing_trials)
        training_array_B.append(training)
        testing_array_B.append(testing)

    training_array_A = np.vstack(training_array_A)
    training_array_B = np.vstack(training_array_B)
    testing_array_A = np.vstack(testing_array_A)
    testing_array_B = np.vstack(testing_array_B)
    axTT = f.add_subplot(gs[18:, 33:])
    sel_training = np.nanmean(training_array_A, 0) - np.nanmean(training_array_B, 0)
    sel_testing = np.nanmean(testing_array_A, 0) - np.nanmean(testing_array_B, 0)
    sns.despine(ax=axTT)
    corr_scatter(sel_training, sel_testing, 'Selectivity (Training)', 'Selectivity (Testing)', ax=axTT, color='k', alpha=0)
    for n in range(dec.n_brains):
        axTT.scatter(sel_training[dec.which_brain == n + 1], sel_testing[dec.which_brain == n + 1].T, color=pltcolors[n], alpha=0.4)

    
    


