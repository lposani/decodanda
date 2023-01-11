import copy

import numpy as np

from .imports import *
from .utilities import *
from .visualize import *


# Main class


class Decodanda:
    def __init__(self,
                 data,
                 conditions,
                 classifier='svc',
                 neural_attr='raster',
                 trial_attr='trial',
                 min_data_per_condition=2,
                 min_trials_per_condition=2,
                 min_activations_per_cell=1,
                 trial_chunk=None,
                 exclude_contiguous_chunks=False,
                 exclude_silent=False,
                 verbose=False,
                 zscore=False,
                 fault_tolerance=False,
                 debug=False,
                 **kwargs
                 ):
        """
        :param data:
        :param conditions:
        :param classifier:
        :param neural_attr:
        :param trial_attr:
        :param min_data_per_condition:
        :param min_trials_per_condition:
        :param min_activations_per_cell:
        :param trial_chunk:
        :param exclude_contiguous_chunks:
        :param exclude_silent:
        :param verbose:
        :param zscore:
        :param fault_tolerance:
        :param debug:
        :param kwargs:
        """
        # casting single session to a list so that it is compatible with all loops below
        if type(data) != list:
            data = [data]

        # handling dictionaries as sessions

        # TODO: change default behavior with dictionaries instead of data structures

        if type(data[0]) == dict:
            dict_sessions = []
            for session in data:
                dict_sessions.append(DictSession(session))
            data = dict_sessions

        # handling discrete dict conditions
        if type(list(conditions.values())[0]) == list:
            conditions = generate_binary_conditions(conditions)

        # setting input parameters
        self.data = data
        self.conditions = conditions
        if classifier == 'svc':
            classifier = LinearSVC(dual=False, C=1.0, class_weight='balanced', max_iter=5000)
        self.classifier = classifier
        self.min_data_per_condition = min_data_per_condition
        self.min_trials_per_condition = min_trials_per_condition
        self.min_activations_per_cell = min_activations_per_cell
        self.verbose = verbose
        self.debug = debug
        self.exclude_silent = exclude_silent
        self.neural_attr = neural_attr
        self.trial_attr = trial_attr
        self.trial_chunk = trial_chunk
        self.exclude_contiguous_trials = exclude_contiguous_chunks

        # setting session(s) data
        self.n_sessions = len(data)
        self.timebin = np.nan
        # TODO: delete all references to timebin
        if hasattr(data[0], 'timebin'):
            self.timebin = data[0].timebin

        self.n_conditions = len(conditions)
        self.max_conditioned_data = 0
        self.min_conditioned_data = 10 ** 6
        self.n_neurons = 0
        self.n_brains = 0
        self.which_brain = []

        # keys and stuff
        self.condition_vectors = generate_binary_words(self.n_conditions)
        self.semantic_keys = list(self.conditions.keys())
        self.semantic_vectors = {string_bool(w): [] for w in generate_binary_words(self.n_conditions)}
        self._generate_semantic_vectors()

        # decoding weights
        if self.n_conditions == 2:
            self.decoding_weights = {key: [] for key in self.semantic_keys + ['XOR']}
            self.decoding_weights_null = {key: [] for key in self.semantic_keys + ['XOR']}
        else:
            self.decoding_weights = {key: [] for key in self.semantic_keys}
            self.decoding_weights_null = {key: [] for key in self.semantic_keys}

        # creating conditioned array with the following structure:
        #   define a condition_vector with boolean values for each semantic condition, es. 100
        #   use this vector as the key for a dictionary
        #   as a value, create a list of neural data for each session conditioned as per key

        #   >>> main object: neural rasters conditioned to semantic vector <<<
        self.conditioned_rasters = {string_bool(w): [] for w in self.condition_vectors}

        # conditioned null model index is the chunk division used for null model shuffles
        self.conditioned_trial_index = {string_bool(w): [] for w in self.condition_vectors}

        #   >>> main part: create conditioned arrays <<< ---------------------------------
        self._divide_data_into_conditions(data)
        #  \ >>> main part: create conditioned arrays <<< --------------------------------

        if zscore:
            self._zscore_activity()

        # raising exceptions
        if self.n_brains == 0:
            if not fault_tolerance:
                raise RuntimeError(
                    "\n[Decodanda] No session passed the minimum data threshold for conditioned arrays.\n\t\t"
                    "Check for mutually-exclusive conditions or try using less restrictive thresholds.")
        else:
            # derived attributes
            self._compute_centroids()

            # null model variables
            self.random_translations = {string_bool(w): [] for w in self.condition_vectors}
            self.subset = np.arange(self.n_neurons)

            self.ordered_conditioned_rasters = {}
            self.ordered_conditioned_trial_index = {}

            for w in self.conditioned_rasters.keys():
                self.ordered_conditioned_rasters[w] = self.conditioned_rasters[w].copy()
                self.ordered_conditioned_trial_index[w] = self.conditioned_trial_index[w].copy()

    # basic decoding functions

    def _train(self, training_raster_A, training_raster_B, label_A, label_B):

        training_labels_A = np.repeat(label_A, training_raster_A.shape[0]).astype(object)
        training_labels_B = np.repeat(label_B, training_raster_B.shape[0]).astype(object)

        training_raster = np.vstack([training_raster_A, training_raster_B])
        training_labels = np.hstack([training_labels_A, training_labels_B])

        self.classifier = sklearn.base.clone(self.classifier)

        training_raster = training_raster[:, self.subset]

        self.classifier.fit(training_raster, training_labels)

    def _test(self, testing_raster_A, testing_raster_B, label_A, label_B):

        testing_labels_A = np.repeat(label_A, testing_raster_A.shape[0]).astype(object)
        testing_labels_B = np.repeat(label_B, testing_raster_B.shape[0]).astype(object)

        testing_raster = np.vstack([testing_raster_A, testing_raster_B])
        testing_labels = np.hstack([testing_labels_A, testing_labels_B])

        testing_raster = testing_raster[:, self.subset]

        if self.debug:
            print("Real labels")
            print(testing_labels)
            print("Predicted labels")
            print(self.classifier.predict(testing_raster))
        performance = self.classifier.score(testing_raster, testing_labels)
        return performance

    def _one_cv_step(self, dic, training_fraction, ndata, shuffled=False, destroy_correlations=False,
                     testing_trials=None):
        """
        :param dic: the dichotomy to be decoded, expressed in a double-list binary format, e.g. [['10', '11'], ['01', '00']]
        :param training_fraction: the fraction of trials used for training.
        :param ndata: the number of population vectors sampled for training and for testing per each condition.
        :param shuffled: if True, population vectors for each condition are sampled in a shuffled way compatible with a null model.
        :param destroy_correlations: if True, correlations within populatin vectors in the sampled training and testing data are destroyed by horizontal shuffling.
        :param testing_trials: if specified, these trials will be used for testing, and the remaining ones for training.
        :return: performance: decoding performance.
        """

        dic_key = self._dic_key(dic)
        set_A = dic[0]
        label_A = ''
        for d in set_A:
            label_A += (self.semantic_vectors[d] + ' ')
        label_A = label_A[:-1]

        set_B = dic[1]
        label_B = ''
        for d in set_B:
            label_B += (self.semantic_vectors[d] + ' ')
        label_B = label_B[:-1]

        training_array_A = []
        training_array_B = []
        testing_array_A = []
        testing_array_B = []

        for d in set_A:
            # TODO: make this function private
            training, testing = sample_training_testing_from_rasters(self.conditioned_rasters[d],
                                                                     ndata,
                                                                     training_fraction,
                                                                     self.conditioned_trial_index[d],
                                                                     debug=self.debug,
                                                                     testing_trials=testing_trials)
            if self.debug:
                plt.title('Condition A')
                print("Sampling for condition A, d=%s" % d)
                print("Conditioned raster mean:")
                print(np.nanmean(self.conditioned_rasters[d][0], 0))

            training_array_A.append(training)
            testing_array_A.append(testing)

        for d in set_B:
            training, testing = sample_training_testing_from_rasters(self.conditioned_rasters[d],
                                                                     ndata,
                                                                     training_fraction,
                                                                     self.conditioned_trial_index[d],
                                                                     debug=self.debug,
                                                                     testing_trials=testing_trials)
            training_array_B.append(training)
            testing_array_B.append(testing)
            if self.debug:
                plt.title('Condition B')
                print("Sampling for condition B, d=%s" % d)
                print("Conditioned raster mean:")
                print(np.nanmean(self.conditioned_rasters[d][0], 0))

        training_array_A = np.vstack(training_array_A)
        training_array_B = np.vstack(training_array_B)
        testing_array_A = np.vstack(testing_array_A)
        testing_array_B = np.vstack(testing_array_B)

        if self.debug:
            selectivity_training = np.nanmean(training_array_A, 0) - np.nanmean(training_array_B, 0)
            selectivity_testing = np.nanmean(testing_array_A, 0) - np.nanmean(testing_array_B, 0)
            corr_scatter(selectivity_training, selectivity_testing, 'Selectivity (training)', 'Selectivity (testing)')

        # TODO: do we need this?
        if destroy_correlations:
            destroy_time_correlations(training_array_A)
            destroy_time_correlations(training_array_B)
            destroy_time_correlations(testing_array_A)
            destroy_time_correlations(testing_array_B)

        # TODO: do we need this?
        # if shuffled:
        #     [np.random.shuffle(x) for x in training_array_A]
        #     [np.random.shuffle(x) for x in training_array_B]
        #     [np.random.shuffle(x) for x in testing_array_A]
        #     [np.random.shuffle(x) for x in testing_array_B]

        self._train(training_array_A, training_array_B, label_A, label_B)

        if hasattr(self.classifier, 'coef_'):
            if dic_key and not shuffled:
                self.decoding_weights[dic_key].append(self.classifier.coef_)
            if dic_key and shuffled:
                # self.decoding_weights_null[dic_key][-1].append(self.classifier.coef_)
                self.decoding_weights_null[dic_key].append(self.classifier.coef_)

        performance = self._test(testing_array_A, testing_array_B, label_A, label_B)

        return performance

    # Dichotomy decoding functions

    def decode_dichotomy(self, dichotomy: Union[str, list], training_fraction: float,
                         cross_validations: int = 10, ndata: Optional[int] = None,
                         shuffled: bool = False, parallel: bool = False,
                         testing_trials: Optional[list] = None, **kwargs) -> list:
        """
        Function that performs cross-validated decoding of a specific dichotomy.

        Parameters
        ----------
            dichotomy:
                The dichotomy to be decoded, expressed in a double-list binary format, e.g. [['10', '11'], ['01', '00']], or as a variable name.
            training_fraction:
                the fraction of trials used for training in each cross-validation fold.
            cross_validations:
                the number of cross-validations.
            ndata:
                the number of data points (population vectors) sampled for training and for testing for each condition.
            shuffled:
                if True, population vectors for each condition are sampled in a random way compatibly with a null model for decoding performance.
            parallel:
                if True, each cross-validation is performed by a dedicated thread (experimental, use with caution).
            testing_trials:
                if specified, data sampled from the specified trial numbers will be used for testing, and the remaining ones for training.

        Returns
        -------
            performances: list of decoding performance values for each cross-validation.

        Note
        ----
        dic can be a string with the same name of the variable specified in the conditions dictionary, or
        a list of two lists, each specifying the conditions used to define the decoded classes in binary notation

        For example, if the data set has two variables `stimulus` :math:`\\in` {-1, 1} and `action` :math:`\\in` {-1, 1}


        >>> dic = 'stimulus'

        is equivalent to

        >>> dic = [['00', '01'], ['10', '11']]

        and

        >>> dic = 'action'

        is equivalent to

        >>> dic = [['00', '10'], ['01, '11]]

        However, not all dichotomies have names (are semantic). For example, the dichotomy

        >>> xor = [['01','10'], ['00', '11']]

        can only be defined in binary notation.

        Note that this function gives you the flexibility to use sub-sets of conditions, for example

        >>> dic = [['10'], ['01']]

        will decode stimulus=1 & action=-1  vs.  stimulus=-1 & action=1


        Example
        -------
        >>> data = generate_synthetic_data(keyA='stimulus', keyB='action')
        >>> dec = Decodanda(data=data, conditions={'stimulus': [-1, 1], 'action': [-1, 1]})
        >>> perfs = dec.decode_dichotomy('stimulus', training_fraction=0.75, cross_validations=10)
        >>> perfs
        [0.82, 0.87, 0.75, ..., 0.77] # 10 values

        """

        # TODO: make these comments into proper doc
        # dic is in the form of a 2xL list, where L is the number of condition vectors in a dichtomy
        # Example: dic = [['10', '11'], ['00', '01']]
        #
        # Decoding works by sampling a balanced amount of patterns from each condition in each class of the dichotomy
        # Each condition is individually divided into training and testing bins

        if type(dichotomy) == str:
            dic = self.dichotomy_from_key(dichotomy)
        else:
            dic = dichotomy
        if ndata is None and self.n_brains == 1:
            ndata = self.max_conditioned_data
        if ndata is None and self.n_brains > 1:
            ndata = max(self.max_conditioned_data, 2 * self.n_neurons)

        if shuffled:
            self._shuffle_conditioned_arrays(dic)

        if self.verbose and not shuffled:
            log_dichotomy(self, dic, ndata, 'Decoding')
            count = tqdm(range(cross_validations))
        else:
            count = range(cross_validations)

        if parallel:
            pool = Pool()
            performances = pool.map(CrossValidator(classifier=self.classifier,
                                                   conditioned_rasters=self.conditioned_rasters,
                                                   conditioned_trial_index=self.conditioned_trial_index,
                                                   dic=dic,
                                                   training_fraction=training_fraction,
                                                   ndata=ndata,
                                                   subset=self.subset,
                                                   semantic_vectors=self.semantic_vectors),
                                    range(cross_validations))

        else:
            performances = np.zeros(cross_validations)
            if self.verbose and not shuffled:
                print('\nLooping over decoding cross validation folds:')
            for i in count:
                performances[i] = self._one_cv_step(dic=dic, training_fraction=training_fraction, ndata=ndata,
                                                    shuffled=shuffled, destroy_correlations=destroy_correlations,
                                                    testing_trials=testing_trials)

        if shuffled:
            self._order_conditioned_rasters()
        return performances

    def CCGP_dichotomy_old(self, dic, ntrials=3, ndata='auto', only_semantic=True, shuffled=False,
                           destroy_correlations=False):

        # TODO: make these comments into proper doc
        # dic is in the form of a 2xL list, where L is the number of condition vectors in a dichtomy
        # Example: dic = [['10', '11'], ['00', '01']]
        #
        # CCGP analysis works by choosing one condition vector from each class of the dichotomies, train over
        # the remaining L-1 vs L-1, and use the two selected condition vectors for testing

        if ndata == 'auto' and self.n_brains == 1:
            ndata = self.max_conditioned_data
        if ndata == 'auto' and self.n_brains > 1:
            ndata = max(self.max_conditioned_data, 2 * self.n_neurons)

        if self.verbose and not shuffled:
            log_dichotomy(self, dic, ndata, 'Cross-condition decoding')

        if shuffled:
            self._rototraslate_conditioned_rasters()
        else:
            self._print('\nLooping over CCGP sampling repetitions:')

        all_performances = []
        if not shuffled and self.verbose:
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
                            label_A += (self.semantic_vectors[ck] + ' ')

                        for ck in training_conditions_B:
                            arr = sample_from_rasters(self.conditioned_rasters[ck], ndata=ndata)
                            training_array_B.append(arr)
                            label_B += (self.semantic_vectors[ck] + ' ')

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

    def CCGP_dichotomy(self, dic, ntrials=3, ndata='auto', only_semantic=True, shuffled=False):
        # TODO: make these comments into proper doc
        # dic is in the form of a 2xL list, where L is the number of condition vectors in a dichtomy
        # Example: dic = [['10', '11'], ['00', '01']]
        #
        # CCGP analysis works by choosing one condition vector from each class of the dichotomies, train over
        # the remaining L-1 vs L-1, and use the two selected condition vectors for testing
        if type(dic) == str:
            dic = self.dichotomy_from_key(dic)

        if ndata == 'auto' and self.n_brains == 1:
            ndata = self.max_conditioned_data
        if ndata == 'auto' and self.n_brains > 1:
            ndata = max(self.max_conditioned_data, 2 * self.n_neurons)

        all_performances = []

        if not shuffled and self.verbose:
            log_dichotomy(self, dic, ndata, 'Cross-condition decoding')
            iterable = tqdm(range(ntrials))
        else:
            iterable = range(1)

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
                            label_A += (self.semantic_vectors[ck] + ' ')

                        for ck in training_conditions_B:
                            arr = sample_from_rasters(self.conditioned_rasters[ck], ndata=ndata)
                            training_array_B.append(arr)
                            label_B += (self.semantic_vectors[ck] + ' ')

                        training_array_A = np.vstack(training_array_A)
                        training_array_B = np.vstack(training_array_B)

                        testing_array_A = sample_from_rasters(self.conditioned_rasters[test_condition_A], ndata=ndata)
                        testing_array_B = sample_from_rasters(self.conditioned_rasters[test_condition_B], ndata=ndata)

                        if shuffled:
                            rotation_A = np.arange(testing_array_A.shape[1]).astype(int)
                            rotation_B = np.arange(testing_array_B.shape[1]).astype(int)
                            np.random.shuffle(rotation_A)
                            np.random.shuffle(rotation_B)
                            testing_array_A = testing_array_A[:, rotation_A]
                            testing_array_B = testing_array_B[:, rotation_A]

                        self._train(training_array_A, training_array_B, label_A, label_B)
                        performance = self._test(testing_array_A, testing_array_B, label_A, label_B)
                        performances.append(performance)

            all_performances.append(np.nanmean(performances))
        return all_performances

    def decode_with_nullmodel(self, dic, training_fraction, cross_validations=10, nshuffles=25, ndata='auto',
                              parallel=False, return_CV=False, destroy_correlations=False, testing_trials=None,
                              plot=False):
        """
        :param dic: the dichotomy to be decoded, expressed in a double-list binary format, e.g. [['10', '11'], ['01', '00']]
        :param training_fraction: the fraction of trials used for training in each cross-validation fold.
        :param cross_validations: the number of cross-validations
        :param nshuffles: the number of null-model iterations. Each iteration reproduces all cross_validations folds in a shuffled way compatible with a null model.
        :param ndata: the number of population vectors sampled for training and for testing per each condition.
        :param parallel: if True, each cross-validation is performed by a dedicated thread (experimental).
        :param return_CV: if True, all cross-validation values are returned.
        :param destroy_correlations: if True, correlations within populatin vectors in the sampled training and testing data are destroyed by horizontal shuffling.
        :param testing_trials: if specified, these trials will be used for testing, and the remaining ones for training.
        :param plot: if True, the results will be visualized.
        :return: performance, null_model_performances.
        """

        d_performances = self.decode_dichotomy(dic, training_fraction, cross_validations, ndata, parallel=parallel,
                                               destroy_correlations=destroy_correlations, testing_trials=testing_trials)
        if return_CV:
            data_performance = d_performances
        else:
            data_performance = np.nanmean(d_performances)

        if self.verbose and nshuffles:
            print(
                "\n[decode_with_nullmodel]\t data <p> = %.2f" % np.nanmean(d_performances))
            print('\n[decode_with_nullmodel]\tLooping over null model shuffles.')
            count = tqdm(range(nshuffles))
        else:
            count = range(nshuffles)

        null_model_performances = np.zeros(nshuffles)

        for n in count:
            performances = self.decode_dichotomy(dic, training_fraction, cross_validations, ndata, shuffled=True,
                                                 parallel=parallel, destroy_correlations=destroy_correlations,
                                                 testing_trials=testing_trials)
            null_model_performances[n] = np.nanmean(performances)
        if plot:
            visualize_decoding(self, dic, d_performances, null_model_performances,
                               training_fraction=training_fraction, ndata=ndata, testing_trials=testing_trials)

        return data_performance, null_model_performances

    def CCGP_with_nullmodel(self, dic, ntrials=5, nshuffles=25, ndata='auto', only_semantic=True, return_CV=False):
        # TODO: write doc
        performances = self.CCGP_dichotomy(dic, ntrials, ndata, only_semantic=only_semantic)

        if return_CV:
            ccgp = performances
        else:
            ccgp = np.nanmean(performances)

        if self.verbose and nshuffles:
            print("\t\t[CCGP_with_nullmodel]\t\t----- Data: <p> = %.2f -----\n" % np.nanmean(performances))
            count = tqdm(range(nshuffles))
        else:
            count = range(nshuffles)

        if return_CV:
            shuffled_ccgp = np.zeros((nshuffles, ntrials))
        else:
            shuffled_ccgp = np.zeros(nshuffles)

        for n in count:
            performances = self.CCGP_dichotomy(dic, 1, ndata, only_semantic, shuffled=True)
            if return_CV:
                shuffled_ccgp[n] = performances
            else:
                shuffled_ccgp[n] = np.nanmean(performances)

        return ccgp, shuffled_ccgp

    # Analysis functions for semantic dichotomies

    def decode(self, training_fraction, cross_validations=10, nshuffles=25, ndata='auto', plot=False, ax=None,
               parallel=False, xor=False, return_CV=False, destroy_correlations=False, testing_trials=None,
               plot_all=False,
               **kwargs):
        # TODO: definitely write doc

        semantic_dics, semantic_keys = self._find_semantic_dichotomies()

        perfs = {}
        perfs_nullmodel = {}
        for key, dic in zip(semantic_keys, semantic_dics):
            if self.verbose:
                print("\nTesting decoding performance for semantic dichotomy: ", key)
            performance, null_model_performances = self.decode_with_nullmodel(
                dic,
                training_fraction,
                cross_validations=cross_validations,
                ndata=ndata,
                nshuffles=nshuffles,
                parallel=parallel,
                return_CV=return_CV,
                destroy_correlations=destroy_correlations,
                testing_trials=testing_trials,
                plot=plot_all)

            perfs[key] = performance
            perfs_nullmodel[key] = null_model_performances

        # TODO: maybe create another function that decodes all non-semantic dychotomies instead of adding XOR here?
        # TODO: or a keyword that decodes all non-semantic as well? And XOR becomes the key for dim=2

        if xor and len(self.conditions) == 2:
            xor_dic = [['01', '10'], ['00', '11']]
            perfs_xor, perfs_null_xor = self.decode_with_nullmodel(dic=xor_dic,
                                                                   training_fraction=training_fraction,
                                                                   cross_validations=cross_validations,
                                                                   nshuffles=nshuffles,
                                                                   parallel=parallel,
                                                                   ndata=ndata,
                                                                   return_CV=return_CV,
                                                                   destroy_correlations=destroy_correlations,
                                                                   testing_trials=testing_trials,
                                                                   plot=plot_all)
            perfs['XOR'] = perfs_xor
            perfs_nullmodel['XOR'] = perfs_null_xor
        if plot:
            if not ax:
                f, ax = plt.subplots(figsize=(0.5+1.8 * len(semantic_dics), 3.5))
            plot_perfs_null_model(perfs, perfs_nullmodel, ylabel='Decoding performance', ax=ax, **kwargs)

        return perfs, perfs_nullmodel

    def CCGP(self, ntrials=5, nshuffles=25, ndata='auto', plot=False, ax=None, only_semantic=True, **kwargs):
        semantic_dics, semantic_keys = self._find_semantic_dichotomies()
        # TODO: definintely write doc

        ccgp = {}
        ccgp_nullmodel = {}
        for key, dic in zip(semantic_keys, semantic_dics):
            if self.verbose:
                print("\nTesting CCGP for semantic dichotomy: ", key)
            data_ccgp, null_ccgps = self.CCGP_with_nullmodel(dic, ntrials, nshuffles, ndata,
                                                             only_semantic=only_semantic)
            ccgp[key] = data_ccgp
            ccgp_nullmodel[key] = null_ccgps

        if plot:
            if not ax:
                f, ax = plt.subplots(figsize=(0.5+1.8 * len(semantic_dics), 3.5))
            plot_perfs_null_model(ccgp, ccgp_nullmodel, ylabel='CCGP', ax=ax, **kwargs)

        return ccgp, ccgp_nullmodel

    def geometrical_analysis(self, training_fraction=0.75, cross_validations=10, nshuffles=10, ndata='auto',
                             visualize=True, z_score_res=False):
        all_dics = generate_dichotomies(self.n_conditions)[1]
        semantic_overlap = []
        dic_name = []

        for i, dic in enumerate(all_dics):
            semantic_overlap.append(semantic_score(dic))
            dic_name.append(str(self._dic_key(dic)))
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
            res, null = self.decode_with_nullmodel(dic,
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
            res, null = self.CCGP_with_nullmodel(dic,
                                                 ntrials=cross_validations,
                                                 nshuffles=nshuffles,
                                                 ndata=ndata,
                                                 only_semantic=False)
            print(i, res)
            CCGP_results.append(res)
            CCGP_null.append(null)

        # plotting
        if visualize:
            if self.n_conditions > 2:
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

        return [all_dics, semantic_overlap], [decoding_results, decoding_null], [CCGP_results, CCGP_null]

    # init utilities

    def _divide_data_into_conditions(self, sessions):
        # TODO: rename sessions into datasets?
        # TODO: make sure conditions don't overlap somehow

        for si, session in enumerate(sessions):

            if self.verbose:
                if hasattr(session, 'name'):
                    print("\t\t[Decodanda]\tbuilding conditioned rasters for session %s" % session.name)
                else:
                    print("\t\t[Decodanda]\tbuilding conditioned rasters for session %u" % si)

            session_conditioned_rasters = {}
            session_conditioned_trial_index = {}

            # exclude inactive neurons
            array = getattr(session, self.neural_attr)
            total_mask = np.zeros(len(array)) > 0

            for condition_vec in self.condition_vectors:
                mask = np.ones(len(array)) > 0
                for i, sk in enumerate(self.semantic_keys):
                    semantic_values = list(self.conditions[sk])
                    mask_i = self.conditions[sk][semantic_values[condition_vec[i]]](session)
                    mask = mask & mask_i
                total_mask = total_mask | mask

            min_activity_mask = np.sum(array[total_mask] > 0, 0) >= self.min_activations_per_cell

            for condition_vec in self.condition_vectors:
                # get the array from the session object
                array = getattr(session, self.neural_attr)
                array = array[:, min_activity_mask]

                # create a mask that becomes more and more restrictive by iterating on semanting conditions
                mask = np.ones(len(array)) > 0
                for i, sk in enumerate(self.semantic_keys):
                    semantic_values = list(self.conditions[sk])
                    mask_i = self.conditions[sk][semantic_values[condition_vec[i]]](session)
                    mask = mask & mask_i

                # select bins conditioned on the semantic behavioural vector
                conditioned_raster = array[mask, :]

                # TODO: fix this nonsense
                # Define trial logic ---
                # chunk: used for cross-validation sampling
                # trial: used for null model, changing labels between trials
                # if trial_chunk_size is specified, chunk and trial are different objects.

                if self.trial_attr is not None:
                    conditioned_trial = getattr(session, self.trial_attr)[mask]
                elif self.trial_chunk is None:
                    if self.verbose:
                        print('[Decodanda]\tUsing contiguous chunks of the same labels as trials.')
                    conditioned_trial = contiguous_chunking(mask)[mask]
                else:
                    conditioned_trial = contiguous_chunking(mask, self.trial_chunk)[mask]

                if self.exclude_contiguous_trials:
                    contiguous_chunks = contiguous_chunking(mask)[mask]
                    nc_mask = non_contiguous_mask(contiguous_chunks, conditioned_trial)
                    conditioned_raster = conditioned_raster[nc_mask, :]
                    conditioned_trial = conditioned_trial[nc_mask]

                # exclude empty time bins (only for binary discrete decoding)
                if self.exclude_silent:
                    active_mask = np.sum(conditioned_raster, 1) > 0
                    conditioned_raster = conditioned_raster[active_mask, :]
                    conditioned_trial = conditioned_trial[active_mask]

                # set the conditioned neural data in the conditioned_rasters dictionary
                session_conditioned_rasters[string_bool(condition_vec)] = conditioned_raster
                session_conditioned_trial_index[string_bool(condition_vec)] = conditioned_trial

                if self.verbose:
                    semantic_vector_string = []
                    for i, sk in enumerate(self.semantic_keys):
                        semantic_values = list(self.conditions[sk])
                        semantic_vector_string.append("%s = %s" % (sk, semantic_values[condition_vec[i]]))
                    semantic_vector_string = ', '.join(semantic_vector_string)
                    print("\t\t\t(%s):\tSelected %u time bin out of %u, divided into %u trials "
                          % (semantic_vector_string, conditioned_raster.shape[0], len(array),
                             len(np.unique(conditioned_trial))))

            session_conditioned_data = [r.shape[0] for r in list(session_conditioned_rasters.values())]
            session_conditioned_trials = [len(np.unique(c)) for c in list(session_conditioned_trial_index.values())]

            self.max_conditioned_data = max([self.max_conditioned_data, np.max(session_conditioned_data)])
            self.min_conditioned_data = min([self.min_conditioned_data, np.min(session_conditioned_data)])

            # if the session has enough data for each condition, append it to the main data dictionary

            if np.min(session_conditioned_data) >= self.min_data_per_condition and \
                    np.min(session_conditioned_trials) >= self.min_trials_per_condition:
                for cv in self.condition_vectors:
                    self.conditioned_rasters[string_bool(cv)].append(session_conditioned_rasters[string_bool(cv)])
                    self.conditioned_trial_index[string_bool(cv)].append(
                        session_conditioned_trial_index[string_bool(cv)])
                if self.verbose:
                    print('\n')
                self.n_brains += 1
                self.n_neurons += list(session_conditioned_rasters.values())[0].shape[1]
                self.which_brain.append(np.ones(list(session_conditioned_rasters.values())[0].shape[1]) * self.n_brains)
            else:
                if self.verbose:
                    print('\t\t\t===> Session discarded for insufficient data.\n')
        if len(self.which_brain):
            self.which_brain = np.hstack(self.which_brain)

    def _find_semantic_dichotomies(self):
        d_keys, dics = generate_dichotomies(self.n_conditions)
        semantic_dics = []
        semantic_keys = []

        for i, dic in enumerate(dics):
            d = [string_bool(x) for x in dic[0]]
            col_sum = np.sum(d, 0)
            if (0 in col_sum) or (len(dic[0]) in col_sum):
                semantic_dics.append(dic)
                semantic_keys.append(self.semantic_keys[np.where(col_sum == len(dic[0]))[0][0]])
        return semantic_dics, semantic_keys

    def _find_nonsemantic_dichotomies(self):
        d_keys, dics = generate_dichotomies(self.n_conditions)
        nonsemantic_dics = []

        for i, dic in enumerate(dics):
            d = [string_bool(x) for x in dic[0]]
            col_sum = np.sum(d, 0)
            if not ((0 in col_sum) or (len(dic[0]) in col_sum)):
                nonsemantic_dics.append(dic)
        return nonsemantic_dics

    def _dic_key(self, dic):
        for i in range(len(dic)):
            d = [string_bool(x) for x in dic[i]]
            col_sum = np.sum(d, 0)
            if len(dic[0]) in col_sum:
                return self.semantic_keys[np.where(col_sum == len(dic[0]))[0][0]]
            elif self.n_conditions == 2:
                return 'XOR'
        return 0

    def dichotomy_from_key(self, key):
        dics, keys = self._find_semantic_dichotomies()
        if key in keys:
            dic = dics[np.where(np.asarray(keys) == key)[0][0]]
        else:
            raise RuntimeError(
                "\n[dichotomy_from_key] The specified key does not correspond to a semantic dichotomy. Check the key value.")

        return dic

    def _generate_semantic_vectors(self):
        for condition_vec in self.condition_vectors:
            semantic_vector = '('
            for i, sk in enumerate(self.semantic_keys):
                semantic_values = list(self.conditions[sk])
                semantic_vector += semantic_values[condition_vec[i]]
            semantic_vector = semantic_vector + ')'
            self.semantic_vectors[string_bool(condition_vec)] = semantic_vector

    def _compute_centroids(self):
        self.centroids = {w: np.hstack([np.nanmean(r, 0) for r in self.conditioned_rasters[w]])
                          for w in self.conditioned_rasters.keys()}

    def _zscore_activity(self):
        keys = [string_bool(w) for w in self.condition_vectors]
        for n in range(self.n_brains):
            n_neurons = self.conditioned_rasters[keys[0]][n].shape[1]
            for i in range(n_neurons):
                r = np.hstack([self.conditioned_rasters[key][n][:, i] for key in keys])
                m = np.nanmean(r)
                std = np.nanstd(r)
                if std:
                    for key in keys:
                        self.conditioned_rasters[key][n][:, i] = (self.conditioned_rasters[key][n][:, i] - m) / std

    def _print(self, string):
        if self.verbose:
            print(string)

    # null model utilities

    def _generate_random_subset(self, n):
        if n < self.n_neurons:
            self.subset = np.random.choice(self.n_neurons, n, replace=False)
        else:
            self.subset = np.arange(self.n_neurons)

    def _reset_random_subset(self):
        self.subset = np.arange(self.n_neurons)

    def _shuffle_conditioned_arrays(self, dic):
        """
        the null model is built by interchanging trials between conditioned arrays that are in different
        dichotomies but have only hamming distance = 1. This ensures that even in the null model the other
        conditions (i.e., the one that do not define the dichotomy), are balanced during sampling.
        So if my dichotomy is [1A, 1B] vs [0A, 0B], I will change trials between 1A and 0A, so that,
        with oversampling, I will then ensure balance between A and B.
        If the dichotomy is not semantic, then I'll probably have to interchange between conditions regardless
        (to be implemented).

        :param dic: The dichotomy to be decoded
        """
        # if the dichotomy is semantic, shuffle between rasters at semantic distance=1
        if self._dic_key(dic):
            set_A = dic[0]
            set_B = dic[1]

            for i in range(len(set_A)):
                for j in range(len(set_B)):
                    test_condition_A = set_A[i]
                    test_condition_B = set_B[j]
                    if hamming(string_bool(test_condition_A), string_bool(test_condition_B)) == 1:
                        for n in range(self.n_brains):
                            # select conditioned rasters
                            arrayA = np.copy(self.conditioned_rasters[test_condition_A][n])
                            arrayB = np.copy(self.conditioned_rasters[test_condition_B][n])

                            # select conditioned trial index
                            trialA = np.copy(self.conditioned_trial_index[test_condition_A][n])
                            trialB = np.copy(self.conditioned_trial_index[test_condition_B][n])

                            n_trials_A = len(np.unique(trialA))
                            n_trials_B = len(np.unique(trialB))

                            # assign randomly trials between the two conditioned rasters, keeping the same
                            # number of trials between the two conditions

                            all_rasters = []
                            all_trials = []

                            for index in np.unique(trialA):
                                all_rasters.append(arrayA[trialA == index, :])
                                all_trials.append(trialA[trialA == index])

                            for index in np.unique(trialB):
                                all_rasters.append(arrayB[trialB == index, :])
                                all_trials.append(trialB[trialB == index])

                            all_trial_index = np.arange(n_trials_A + n_trials_B).astype(int)
                            np.random.shuffle(all_trial_index)

                            new_rasters_A = [all_rasters[iA] for iA in all_trial_index[:n_trials_A]]
                            new_rasters_B = [all_rasters[iB] for iB in all_trial_index[n_trials_A:]]

                            new_trials_A = [all_trials[iA] for iA in all_trial_index[:n_trials_A]]
                            new_trials_B = [all_trials[iB] for iB in all_trial_index[n_trials_A:]]

                            self.conditioned_rasters[test_condition_A][n] = np.vstack(new_rasters_A)
                            self.conditioned_rasters[test_condition_B][n] = np.vstack(new_rasters_B)

                            self.conditioned_trial_index[test_condition_A][n] = np.hstack(new_trials_A)
                            self.conditioned_trial_index[test_condition_B][n] = np.hstack(new_trials_B)

        else:
            dics, keys = self._find_semantic_dichotomies()
            for di in dics:
                self._shuffle_conditioned_arrays(di)

        if not self._check_trial_availability():  # if the trial distribution is not cross validatable, redo the shuffling
            print("Note: re-shuffling arrays")
            self._order_conditioned_rasters()
            self._shuffle_conditioned_arrays(dic)

    def _rototraslate_conditioned_rasters(self):
        # DEPCRECATED

        for i in range(self.n_brains):
            # brain_means = np.vstack([np.nanmean(self.conditioned_rasters[key][i], 0) for key in self.conditioned_rasters.keys()])
            # mean_centroid = np.nanmean(brain_means, axis=0)
            for w in self.conditioned_rasters.keys():
                raster = self.conditioned_rasters[w][i]
                rotation = np.arange(raster.shape[1]).astype(int)
                np.random.shuffle(rotation)
                raster = raster[:, rotation]
                # mean = np.nanmean(raster, 0)
                # randomdir = np.random.rand()-0.5
                # randomdir = randomdir/np.sqrt(np.dot(randomdir, randomdir))
                # vector_from_mean_centroid = mean - mean_centroid
                # distance_from_mean_centroid = np.sqrt(np.dot(vector_from_mean_centroid, vector_from_mean_centroid))
                # raster = raster - vector_from_mean_centroid + randomdir*distance_from_mean_centroid
                self.conditioned_rasters[w][i] = raster

    def _order_conditioned_rasters(self):
        for w in self.conditioned_rasters.keys():
            self.conditioned_rasters[w] = self.ordered_conditioned_rasters[w].copy()
            self.conditioned_trial_index[w] = self.ordered_conditioned_trial_index[w].copy()

    def _check_trial_availability(self):
        if self.debug:
            print('\nCheck trial availability')
        for k in self.conditioned_trial_index:
            for i, ti in enumerate(self.conditioned_trial_index[k]):
                if self.debug:
                    print(k, 'raster %u:' % i, np.unique(ti).shape[0])
                    print(ti)
                if np.unique(ti).shape[0] < 2:
                    return False
        return True


# Utilities

# TODO: for all utilities: write doc or make private

def check_session_requirements(session, conditions, **decodanda_params):
    d = Decodanda(session, conditions, fault_tolerance=True, **decodanda_params)
    if d.n_brains:
        return True
    else:
        return False


def check_requirements_two_conditions(sessions, conditions_1, conditions_2, **decodanda_params):
    good_sessions = []
    for s in sessions:
        if check_session_requirements(s, conditions_1, **decodanda_params) and check_session_requirements(s, conditions_2,
                                                                                                          **decodanda_params):
            good_sessions.append(s)
    return good_sessions


def balance_decodandas(ds):
    for i in range(len(ds)):
        for j in range(i + 1, len(ds)):
            balance_two_decodandas(ds[i], ds[j])


def balance_two_decodandas(d1, d2, sampling_strategy='random'):
    assert d1.n_brains == d2.n_brains, "The two decodandas do not have the same number of brains."
    assert d1.n_conditions == d2.n_conditions, "The two decodanda do not have the same number of semantic conditions."

    n_brains = d1.n_brains
    n_conditioned_rasters = len(list(d1.conditioned_rasters.values()))

    for n in range(n_brains):
        for i in range(n_conditioned_rasters):
            t1 = list(d1.conditioned_rasters.values())[i][n].shape[0]
            t2 = list(d2.conditioned_rasters.values())[i][n].shape[0]
            t = min(t1, t2)

            if t1 > t2:
                if sampling_strategy == 'random':
                    sampling = np.random.choice(t1, t2, replace=False)
                if sampling_strategy == 'ordered':
                    sampling = np.arange(t2, dtype=int)
                list(d1.conditioned_rasters.values())[i][n] = list(d1.conditioned_rasters.values())[i][n][sampling, :]
                list(d1.conditioned_trial_index.values())[i][n] = list(d1.conditioned_trial_index.values())[i][n][
                    sampling]

            if t2 > t1:
                if sampling_strategy == 'random':
                    sampling = np.random.choice(t2, t1, replace=False)
                if sampling_strategy == 'ordered':
                    sampling = np.arange(t1, dtype=int)
                list(d2.conditioned_rasters.values())[i][n] = list(d2.conditioned_rasters.values())[i][n][sampling, :]
                list(d2.conditioned_trial_index.values())[i][n] = list(d2.conditioned_trial_index.values())[i][n][
                    sampling]

            print("Balancing data for d1: %u, d2: %u - now d1: %u, d2: %u" % (
                t1, t2, list(d1.conditioned_rasters.values())[i][n].shape[0],
                list(d2.conditioned_rasters.values())[i][n].shape[0]))

    for w in d1.conditioned_rasters.keys():
        d1.ordered_conditioned_rasters[w] = d1.conditioned_rasters[w].copy()
        d1.ordered_conditioned_trial_index[w] = d1.conditioned_trial_index[w].copy()
        d2.ordered_conditioned_rasters[w] = d2.conditioned_rasters[w].copy()
        d2.ordered_conditioned_trial_index[w] = d2.conditioned_trial_index[w].copy()

    print("\n")


def generate_binary_condition(var_key, value1, value2, key1=None, key2=None, var_key_plot=None):
    if key1 is None:
        key1 = '%s' % value1
    if key2 is None:
        key2 = '%s' % value2
    if var_key_plot is None:
        var_key_plot = var_key

    conditions = {
        var_key_plot: {
            key1: lambda d, x=value1: d[var_key] == x,
            key2: lambda d, x=value2: d[var_key] == x,
        }
    }

    return conditions


def generate_binary_conditions(discrete_dict):
    conditions = {}
    for key in discrete_dict.keys():
        conditions[key] = {
            '%s' % discrete_dict[key][0]: lambda d, k=key: getattr(d, k) == discrete_dict[k][0],
            '%s' % discrete_dict[key][1]: lambda d, k=key: getattr(d, k) == discrete_dict[k][1],
        }
    return conditions


class NullmodelIterator(object):  # necessary for parallelization of null model iterations
    def __init__(self, data, conditions, decodanda_params, analysis_params):
        self.data = data
        self.conditions = conditions
        self.decodanda_params = decodanda_params
        self.analysis_params = analysis_params

    def __call__(self, i):
        self.i = i
        self.randomstate = RandomState(i)
        dec = Decodanda(data=self.data, conditions=self.conditions, **self.decodanda_params)
        semantic_dics, semantic_keys = dec._find_semantic_dichotomies()
        if 'XOR' in self.analysis_params.keys():
            if self.analysis_params['XOR'] and len(self.conditions) == 2:
                semantic_dics.append([['01', '10'], ['00', '11']])
                semantic_keys.append('XOR')

        perfs = {}
        for key, dic in zip(semantic_keys, semantic_dics):
            if dec.verbose:
                print("\nTesting null decoding performance for semantic dichotomy: ", key)
            dec._shuffle_conditioned_arrays(dic)
            performance = dec.decode_dichotomy(dic, **self.analysis_params)
            perfs[key] = np.nanmean(performance)
            dec._order_conditioned_rasters()

        return perfs


def decoding_analysis(data, conditions, decodanda_params, analysis_params, parallel=False, plot=False, ax=None):
    an_params = copy.deepcopy(analysis_params)

    if parallel:
        # Data
        null_iterations = an_params['nshuffles']
        an_params['nshuffles'] = 0
        performances, _ = Decodanda(data, conditions, **decodanda_params).decode(**an_params)

        # Null
        del an_params['nshuffles']
        pool = Pool()
        null_performances = pool.map(NullmodelIterator(data, conditions, decodanda_params, an_params),
                                     range(null_iterations))
        null = {key: np.stack([p[key] for p in null_performances]) for key in null_performances[0].keys()}
    else:
        performances, null = Decodanda(data, conditions, **decodanda_params).decode(**an_params)

    if plot:
        plot_perfs_null_model(performances, null, ax=ax, ptype='zscore')
    return performances, null
