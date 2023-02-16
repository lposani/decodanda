from decodanda import *


def geometrical_analysis(dec,
                         training_fraction: float = 0.75,
                         cross_validations: int = 10,
                         nshuffles: int = 10,
                         ndata: Optional[int] = None,
                         visualize=True):
    """
    This function performs a balanced decoding analysis for each possible dichotomy, and
    plots the result sorted by a semantic score that tells how close each dichotomy is to
    any of the specified variables. A semantic dichotomy has ``semantic_score = 1``, the
    XOR dichotomy has ``semantic_score = 0``.

    Parameters
    ----------
    dec
        The Decodanda object
    training_fraction:
        the fraction of trials used for training in each cross-validation fold.
    cross_validations:
        the number of cross-validations.
    nshuffles:
        the number of null-model iterations of the decoding procedure.
    ndata:
        the number of data points (population vectors) sampled for training and for testing for each condition.
    visualize:
        if ``True``, the decoding results are shown in a figure.


    Returns
    -------
    dichotomies_data:
        Two lists, one containing all the dichotomies in binary notation
        and one containing the corresponding semantic score.
    decoding_data:
        Two dictionaries, one containing the decoding performances for all dichotomies
        and one containing all the corresponding lists of null model performances.
    CCGP_data:
        Two dictionaries, one containing the CCGP values for all dichotomies
        and one containing all the corresponding lists of null model values.
    """

    all_dics = generate_dichotomies(dec.n_conditions)[1]
    semantic_overlap = []
    dic_name = []

    for i, dic in enumerate(all_dics):
        semantic_overlap.append(semantic_score(dic))
        dic_name.append(str(dec._dic_key(dic)))
    semantic_overlap = np.asarray(semantic_overlap)

    # sorting dichotomies wrt semantic overlap
    dic_name = np.asarray(dic_name)[np.argsort(semantic_overlap)[::-1]]
    all_dics = list(np.asarray(all_dics)[np.argsort(semantic_overlap)[::-1]])
    semantic_overlap = semantic_overlap[np.argsort(semantic_overlap)[::-1]]
    semantic_overlap = (semantic_overlap - np.min(semantic_overlap)) / (
            np.max(semantic_overlap) - np.min(semantic_overlap))

    # decoding all dichotomies
    decoding_results = []
    decoding_null = []
    for i, dic in enumerate(all_dics):
        res, null = dec.decode_with_nullmodel(dic,
                                              training_fraction=training_fraction,
                                              cross_validations=cross_validations,
                                              nshuffles=nshuffles,
                                              ndata=ndata)
        print(i, res)
        decoding_results.append(res)
        decoding_null.append(null)

    # CCGP all dichotomies
    CCGP_results = []
    CCGP_null = []
    for i, dic in enumerate(all_dics):
        print(dic)
        res, null = dec.CCGP_with_nullmodel(dic,
                                            nshuffles=nshuffles,
                                            ndata=ndata,
                                            max_semantic_dist=dec.n_conditions)
        print(i, res)
        CCGP_results.append(res)
        CCGP_null.append(null)

    # plotting
    if visualize:
        if dec.n_conditions > 2:
            f, axs = plt.subplots(2, 1, figsize=(6, 6))
            axs[0].set_xlabel('Dichotomy (ordered by semantic score)')
            axs[1].set_xlabel('Dichotomy (ordered by semantic score)')
        else:
            f, axs = plt.subplots(1, 2, figsize=(6, 3.5))
            axs[0].set_xlabel('Dichotomy')
            axs[1].set_xlabel('Dichotomy')
            axs[0].set_xlim([-0.5, 2.5])
            axs[1].set_xlim([-0.5, 2.5])

        axs[0].set_ylabel('Decoding Performance')
        axs[1].set_ylabel('CCGP')
        axs[0].axhline([0.5], color='k', linestyle='--', alpha=0.5)
        axs[1].axhline([0.5], color='k', linestyle='--', alpha=0.5)
        axs[0].set_xticks([])
        axs[1].set_xticks([])
        axs[0].set_ylim([0, 1.05])
        axs[1].set_ylim([0, 1.05])
        sns.despine(f)

        # visualize Decoding
        for i in range(len(all_dics)):
            if z_pval(decoding_results[i], decoding_null[i])[1] < 0.01:
                axs[0].scatter(i, decoding_results[i], marker='o',
                               color=cm.cool(int(semantic_overlap[i] * 255)), edgecolor='k',
                               s=110, linewidth=2, linestyle='-')
            elif z_pval(decoding_results[i], decoding_null[i])[1] < 0.05:
                axs[0].scatter(i, decoding_results[i], marker='o',
                               color=cm.cool(int(semantic_overlap[i] * 255)), edgecolor='k',
                               s=110, linewidth=2, linestyle='--')
            elif z_pval(decoding_results[i], decoding_null[i])[1] > 0.05:
                axs[0].scatter(i, decoding_results[i], marker='o',
                               color=cm.cool(int(semantic_overlap[i] * 255)), edgecolor='k',
                               s=110, linewidth=2, linestyle='dotted')
            axs[0].errorbar(i, np.nanmean(decoding_null[i]), np.nanstd(decoding_null[i]), color='k', alpha=0.3)

            if dic_name[i] != '0':
                axs[0].text(i, decoding_results[i] + 0.08, dic_name[i], rotation=90, fontsize=6, color='k',
                            ha='center', fontweight='bold')
        # visualize CCGP

        for i in range(len(all_dics)):
            if z_pval(CCGP_results[i], CCGP_null[i])[1] < 0.01:
                axs[1].scatter(i, CCGP_results[i], marker='s',
                               color=cm.cool(int(semantic_overlap[i] * 255)), edgecolor='k',
                               s=110, linewidth=2, linestyle='-')
            elif z_pval(CCGP_results[i], CCGP_null[i])[1] < 0.05:
                axs[1].scatter(i, CCGP_results[i], marker='s',
                               color=cm.cool(int(semantic_overlap[i] * 255)), edgecolor='k',
                               s=110, linewidth=2, linestyle='--')
            elif z_pval(CCGP_results[i], CCGP_null[i])[1] > 0.05:
                axs[1].scatter(i, CCGP_results[i], marker='s',
                               color=cm.cool(int(semantic_overlap[i] * 255)), edgecolor='k',
                               s=110, linewidth=2, linestyle='dotted')

            axs[1].errorbar(i, np.nanmean(CCGP_null[i]), np.nanstd(CCGP_null[i]), color='k', alpha=0.3)

            if dic_name[i] != '0':
                axs[1].text(i, CCGP_results[i] + 0.08, dic_name[i], rotation=90, fontsize=6, color='k',
                            ha='center', fontweight='bold')

    dichotomies_data = [all_dics, semantic_overlap]
    decoding_data = [decoding_results, decoding_null]
    CCGP_data = [CCGP_results, CCGP_null]

    return dichotomies_data, decoding_data, CCGP_data
