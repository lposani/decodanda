#     Copyright (C) 2023  Lorenzo Posani
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#

import copy
from typing import Tuple, Union

import numpy as np
import scipy.stats.stats
from numpy import ndarray
from .imports import *
from .utilities import (CrossValidator, DictSession, contiguous_chunking,
                        cosine, generate_binary_words, generate_dichotomies,
                        hamming, log_dichotomy, non_contiguous_mask,
                        sample_from_rasters,
                        sample_training_testing_from_rasters, semantic_score,
                        string_bool, z_pval)
from .visualize import (corr_scatter, plot_perfs_null_model,
                        visualize_decoding, visualize_PCA)

# Main class

class Decodanda:
    def __init__(self,
                 data: Union[list, dict],
                 conditions: dict,
                 classifier: any = 'svc',
                 neural_attr: str = 'raster',
                 trial_attr: str = 'trial',
                 squeeze_trials: bool = False,
                 min_data_per_condition: int = 2,
                 min_trials_per_condition: int = 2,
                 min_activations_per_cell: int = 1,
                 trial_chunk: Optional[int] = None,
                 exclude_contiguous_chunks: bool = False,
                 exclude_silent: bool = False,
                 verbose: bool = False,
                 zscore: bool = False,
                 fault_tolerance: bool = False,
                 debug: bool = False,
                 **kwargs
                 ):

        """
        Main class that implements the decoding pipelines with built-in best practices.

        It works by separating the input data into all possible conditions - defined as specific
        combinations of variable values - and sampling data points from these conditions
        according to the specific decoding problem.

        Parameters
        ----------
        data
            A dictionary or a list of dictionaries each containing
            (1) the neural data (2) a set of variables that we want to decode from the neural data
            (3) a trial number. See the ``Data Structure`` section for more details.
            If a list is passed, the analyses will be performed on the pseudo-population built by pooling
            all the data sets in the list.

        conditions
            A dictionary that specifies which values for which variables of `data` we want to decode.
            See the ``Data Structure`` section for more details.

        classifier
            The classifier used for all decoding analyses. Default: ``sklearn.svm.LinearSVC``.

        neural_attr
            The key under which the neural features are stored in the ``data`` dictionary.

        trial_attr
            The key under which the trial numbers are stored in the ``data`` dictionary.
            Each different trial is considered as an independent sample to be used in
            during cross validation.
            If ``None``, trials are defined as consecutive bouts of data in time
            where all the variables have a constant value.

        squeeze_trials
            If True, all population vectors corresponding to the same trial number for the same
            condition will be squeezed into a single average activity vector.

        min_data_per_condition
            The minimum number of data points per each condition, defined as a specific
            combination of values of all variables in the ``conditions`` dictionary,
            that a data set needs to have to be included in the analysis.

        min_trials_per_condition
            The minimum number of unique trial numbers per each condition, defined as a specific
            combination of values of all variables in the ``conditions`` dictionary,
            that a data set needs to have to be included in the analysis.

        min_activations_per_cell
            The minimum number of non-zero bins that single neurons / features need to have to be
            included into the analysis.

        trial_chunk
            Only used when ``trial_attr=None``. The maximum number of consecutive data points
            within the same bout. Bouts longer than ``trial_chunk`` data points are split into
            different trials.

        exclude_contiguous_chunks
            Only used when ``trial_attr=None`` and ``trial_chunks != None``. Discards every second trial
            that has the same value of all variables as the previous one. It can be useful to avoid
            decoding temporal artifacts when there are long auto-correlation times in the neural
            activations.

        exclude_silent
            If ``True``, all silent population vectors (only zeros) are excluded from the analysis.

        verbose
            If ``True``, most operations and analysis results are logged in standard output.

        zscore
            If ``True``, neural features are z-scored before being separated into conditions.

        fault_tolerance
            If ``True``, the constructor raises a warning instead of an error if no data set
            passes the inclusion criteria specified by ``min_data_per_condition`` and ``min_trials_per_condition``.

        debug
            If ``True``, operations are super verbose. Do not use unless you are developing.


        Data structure
        --------------
        Decodanda works with datasets organized into Python dictionaries.
        For ``N`` recorded neurons and ``T`` trials (or time bins), the data dictionary must contain:

        1. a ``TxN`` array, under the ``raster`` key
            This is the set of features we use to decode. Can be continuous (e.g., calcium fluorescence) or discrete (e.g., spikes) values.

        2. a ``Tx1`` array specifying a ``trial`` number
            This array will define the subdivisions for cross validation: trials (or time bins) that share the
            same ```trial``` value will always go together in either training or testing samples.

        3. a ``Tx1`` array for each variable we want to decode
            Each value will be used as a label for the ``raster`` feature. Make sure these arrays are
            synchronized with the ``raster`` array.


        Say we have a data set with N=50 neurons, T=800 time bins divided into 80 trials, where two experimental
        variables are specified ``stimulus`` and ``action``.
        A properly-formatted data set would look like this:

        >>> data = {
        >>>     'raster': [[0, 1, ..., 0], ..., [0, 2, ..., 1]],     # <800x50 array>, neural activations
        >>>     'stimulus': ['A', 'A', 'B', ..., 'B'],               # <800x1 array>, values of the stimulus variable
        >>>     'action': ['left', 'left', 'none', ..., 'left'],    # <800x1 array>, values of the action variable
        >>>     'trial':  [1, 1, 1, ..., 2, 2, 2, ..., 80, 80, 80],  # <800x1 array>, trial number, 80 unique numbers
        >>> }

        The ``conditions`` dictionary is used to specify which variables - out of
        all the keywords in the ``data`` dictionary, and which and values - out of
        all possible values of each specified variable - we want to decode.

        It has to be in the form ``{key: [value1, value2]}``:

        >>> conditions = {
        >>>     'stimulus': ['A', 'B'],
        >>>     'action': ['left', 'right']
        >>> }

        If more than one variable is specified, `Decodanda` will balance all
        conditions during each decoding analysis to disentangle
        the variables and avoid confounding correlations.


        Examples
        --------

        Using the data set defined above:

        >>> from decodanda import Decodanda
        >>>
        >>> dec = Decodanda(
        >>>         data=data,
        >>>         conditions=conditions
        >>>         verbose=True)
        >>>
        [Decodanda]	building conditioned rasters for session 0
                    (stimulus = A, action = left):	Selected 150 time bin out of 800, divided into 15 trials
                    (stimulus = A, action = right):	Selected 210 time bin out of 800, divided into 21 trials
                    (stimulus = B, action = left):	Selected 210 time bin out of 800, divided into 21 trials
                    (stimulus = B, action = right):	Selected 230 time bin out of 800, divided into 23 trials


        The constructor divides the data into conditions using the ``stimulus`` and ``action`` values
        and stores them in the ``self.conditioned_rasters`` object.
        This condition structure is the basis for all the balanced decoding analyses.

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

        # check whether conditions are binary
        for key in conditions:
            if len(conditions[key]) != 2:
                raise RuntimeError(
                    f"\n[Decodanda] In this version of Decodanda, variables should be binary\n "
                    f"Variable {key} has {len(conditions[key])} values. Please check the conditions dictionary.")

        # handling discrete dict conditions
        if type(list(conditions.values())[0]) == list:
            conditions = _generate_binary_conditions(conditions)

        # setting input parameters
        self.data = data
        self.conditions = conditions
        if classifier == 'svc':
            classifier = LinearSVC(dual=False, C=1.0, class_weight='balanced', max_iter=5000)
        self.classifier = classifier

        # private params
        self._min_data_per_condition = min_data_per_condition
        self._min_trials_per_condition = min_trials_per_condition
        self._min_activations_per_cell = min_activations_per_cell
        self._verbose = verbose
        self._debug = debug
        self._zscore = zscore
        self._exclude_silent = exclude_silent
        self._neural_attr = neural_attr
        self._trial_attr = trial_attr
        self._trial_chunk = trial_chunk
        self._exclude_contiguous_trials = exclude_contiguous_chunks
        self._trial_average = squeeze_trials

        # setting session(s) data
        self.n_sessions = len(data)
        self.n_conditions = len(conditions)
        self._max_conditioned_data = 0
        self._min_conditioned_data = 10 ** 6
        self.n_neurons = 0
        self.n_brains = 0
        self.which_brain = []

        # keys and stuff
        self._condition_vectors = generate_binary_words(self.n_conditions)  # TODO change this for multimodal
        self._semantic_keys = list(self.conditions.keys())
        self._semantic_vectors = {string_bool(w): [] for w in generate_binary_words(self.n_conditions)}
        self._generate_semantic_vectors()

        # decoding weights
        self.decoding_weights = {}
        self.decoding_weights_null = {}

        # creating conditioned array with the following structure:
        #   define a condition_vector with boolean values for each semantic condition, es. 100
        #   use this vector as the key for a dictionary
        #   as a value, create a list of neural data for each session conditioned as per key

        #   >>> main object: neural rasters conditioned to semantic vector <<<
        self.conditioned_rasters = {string_bool(w): [] for w in self._condition_vectors}

        # conditioned null model index is the chunk division used for null model shuffles
        self.conditioned_trial_index = {string_bool(w): [] for w in self._condition_vectors}

        #   >>> main part: create conditioned arrays <<< ---------------------------------
        self._divide_data_into_conditions(data)
        #  \ >>> main part: create conditioned arrays <<< --------------------------------

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
            self.random_translations = {string_bool(w): [] for w in self._condition_vectors}
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

        if self._debug:
            print("Real labels")
            print(testing_labels)
            print("Predicted labels")
            print(self.classifier.predict(testing_raster))
        performance = self.classifier.score(testing_raster, testing_labels)
        return performance

    def _one_cv_step(self, dic, training_fraction, ndata, shuffled=False, testing_trials=None, dic_key=None):
        if dic_key is None:
            dic_key = self._dic_key(dic)

        set_A = dic[0]
        label_A = ''
        for d in set_A:
            label_A += (self._semantic_vectors[d] + ' ')
        label_A = label_A[:-1]

        set_B = dic[1]
        label_B = ''
        for d in set_B:
            label_B += (self._semantic_vectors[d] + ' ')
        label_B = label_B[:-1]

        training_array_A = []
        training_array_B = []
        testing_array_A = []
        testing_array_B = []

        # allow for unbalanced dichotomies
        n_conditions_A = float(len(dic[0]))
        n_conditions_B = float(len(dic[1]))
        fraction = n_conditions_A / n_conditions_B

        for d in set_A:
            training, testing = sample_training_testing_from_rasters(self.conditioned_rasters[d],
                                                                     int(ndata / fraction),
                                                                     training_fraction,
                                                                     self.conditioned_trial_index[d],
                                                                     debug=self._debug,
                                                                     testing_trials=testing_trials)
            if self._debug:
                plt.title('Condition A')
                print("Sampling for condition A, d=%s" % d)
                print("Conditioned raster mean:")
                print(np.nanmean(self.conditioned_rasters[d][0], 0))

            training_array_A.append(training)
            testing_array_A.append(testing)

        for d in set_B:
            training, testing = sample_training_testing_from_rasters(self.conditioned_rasters[d],
                                                                     int(ndata),
                                                                     training_fraction,
                                                                     self.conditioned_trial_index[d],
                                                                     debug=self._debug,
                                                                     testing_trials=testing_trials)
            training_array_B.append(training)
            testing_array_B.append(testing)
            if self._debug:
                plt.title('Condition B')
                print("Sampling for condition B, d=%s" % d)
                print("Conditioned raster mean:")
                print(np.nanmean(self.conditioned_rasters[d][0], 0))

        training_array_A = np.vstack(training_array_A)
        training_array_B = np.vstack(training_array_B)
        testing_array_A = np.vstack(testing_array_A)
        testing_array_B = np.vstack(testing_array_B)

        if self._debug:
            selectivity_training = np.nanmean(training_array_A, 0) - np.nanmean(training_array_B, 0)
            selectivity_testing = np.nanmean(testing_array_A, 0) - np.nanmean(testing_array_B, 0)
            corr_scatter(selectivity_training, selectivity_testing, 'Selectivity (training)', 'Selectivity (testing)')

        if self._zscore:
            big_raster = np.vstack([training_array_A, training_array_B])  # z-scoring using the training data
            big_mean = np.nanmean(big_raster, 0)
            big_std = np.nanstd(big_raster, 0)
            big_std[big_std == 0] = np.inf
            training_array_A = (training_array_A - big_mean) / big_std
            training_array_B = (training_array_B - big_mean) / big_std
            testing_array_A = (testing_array_A - big_mean) / big_std
            testing_array_B = (testing_array_B - big_mean) / big_std

        self._train(training_array_A, training_array_B, label_A, label_B)

        if hasattr(self.classifier, 'coef_'):
            if dic_key and not shuffled:
                if dic_key not in self.decoding_weights.keys():
                    self.decoding_weights[dic_key] = []
                self.decoding_weights[dic_key].append(self.classifier.coef_)
            if dic_key and shuffled:
                if dic_key not in self.decoding_weights_null.keys():
                    self.decoding_weights_null[dic_key] = []
                self.decoding_weights_null[dic_key].append(self.classifier.coef_)

        performance = self._test(testing_array_A, testing_array_B, label_A, label_B)

        return performance

    def _one_X_cv_step(self, dic1, dic2, training_fraction, ndata, shuffled=False):
        # Training rasters
        training_set_A = dic1[0]
        training_set_B = dic1[1]
        testing_set_A = dic2[0]
        testing_set_B = dic2[1]

        training_array_A = []
        training_array_B = []
        testing_array_A = []
        testing_array_B = []

        # allow for unbalanced dichotomies
        n_conditions_A = float(len(dic1[0]))
        n_conditions_B = float(len(dic1[1]))
        fraction = n_conditions_A / n_conditions_B

        for d in training_set_A:
            training, testing = sample_training_testing_from_rasters(self.conditioned_rasters[d],
                                                                     int(ndata / fraction),
                                                                     training_fraction,
                                                                     self.conditioned_trial_index[d],
                                                                     debug=self._debug)
            if self._debug:
                plt.title('Condition A')
                print("Sampling for condition A, d=%s" % d)
                print("Conditioned raster mean:")
                print(np.nanmean(self.conditioned_rasters[d][0], 0))

            training_array_A.append(training)
            if d in testing_set_A:
                testing_array_A.append(testing)
            elif d in testing_set_B:
                testing_array_B.append(testing)

        for d in training_set_B:
            training, testing = sample_training_testing_from_rasters(self.conditioned_rasters[d],
                                                                     int(ndata),
                                                                     training_fraction,
                                                                     self.conditioned_trial_index[d],
                                                                     debug=self._debug)
            training_array_B.append(training)
            if d in testing_set_A:
                testing_array_A.append(testing)
            elif d in testing_set_B:
                testing_array_B.append(testing)

        training_array_A = np.vstack(training_array_A)
        training_array_B = np.vstack(training_array_B)
        testing_array_A = np.vstack(testing_array_A)
        testing_array_B = np.vstack(testing_array_B)

        if self._debug:
            selectivity_training = np.nanmean(training_array_A, 0) - np.nanmean(training_array_B, 0)
            selectivity_testing = np.nanmean(testing_array_A, 0) - np.nanmean(testing_array_B, 0)
            corr_scatter(selectivity_training, selectivity_testing, 'Selectivity (training)', 'Selectivity (testing)')

        if self._zscore:
            big_raster = np.vstack([training_array_A, training_array_B])  # z-scoring using the training data
            big_mean = np.nanmean(big_raster, 0)
            big_std = np.nanstd(big_raster, 0)
            big_std[big_std == 0] = np.inf
            training_array_A = (training_array_A - big_mean) / big_std
            training_array_B = (training_array_B - big_mean) / big_std
            testing_array_A = (testing_array_A - big_mean) / big_std
            testing_array_B = (testing_array_B - big_mean) / big_std

        self._train(training_array_A, training_array_B, 'A', 'B')

        performance = self._test(testing_array_A, testing_array_B, 'A', 'B')

        return performance

    # Sampling functions

    def balanced_resample(self, condition_names=False, ndata=None, z_score=None, min_ar=0):
        """

        Parameters
        ----------
        condition_names: if True, verbose names for conditions are used, otherwise a binary notation is used. Default: False.

        ndata: optional, number of resampled activity vectors per condition. If not specified,
        the maximum number of activity vectors across all conditions is used.

        z_score: if True, the resampled rasters are z-scored with respect to all the conditions.

        min_ar: neurons below a minimum activity rate (fraction of bins with non-zero activity) threshold specified
        by the ``min_ar`` parameter will be excluded from the sampled data.

        Returns
        -------
        balanced resampled rasters
        """
        if ndata is None:
            ndata = self._max_conditioned_data
        if z_score is None:
            z_score = self._zscore

        resampled_rasters = {}
        for key in self.conditioned_rasters:
            if condition_names:
                condition_key = self._semantic_vectors[key]
            else:
                condition_key = key
            resampled_rasters[condition_key] = []
            for n in range(self.n_brains):
                x = self.conditioned_rasters[key][n]
                sampling_index = np.random.randint(0, x.shape[0], ndata)
                resampled_rasters[condition_key].append(x[sampling_index])
            resampled_rasters[condition_key] = np.hstack(resampled_rasters[condition_key])

        if z_score:
            for i in range(self.n_neurons):
                big_x = np.hstack([r[:, i] for r in resampled_rasters.values()])
                bigmean = np.nanmean(big_x)
                bigstd = np.nanstd(big_x)
                for key in resampled_rasters:
                    resampled_rasters[key][:, i] = (resampled_rasters[key][:, i] - bigmean) / bigstd

        if min_ar:
            X = np.vstack([resampled_rasters[key] for key in resampled_rasters])
            activityrate = np.nanmean(X > 0, 0)
            for key in resampled_rasters:
                resampled_rasters[key] = resampled_rasters[key][:, activityrate > min_ar]

        return resampled_rasters

    def split_resample(self, fraction=0.5, condition_names=False, ndata=None, z_score=None, min_ar=0):
        """

        Parameters ----------

        fraction: the fraction of trials used to sample from to fill the first data set (
        raster_A). The remaining fraction (1-``fraction``) is used to sample the second data set (raster_B)

        condition_names: if True, verbose names for conditions are used, otherwise a binary notation is used. Default: False.

        ndata: optional, number of resampled activity vectors per condition. If not specified,
        the maximum number of activity vectors across all conditions is used.

        z_score: if True, the resampled rasters are z-scored with respect to all the conditions.

        min_ar: neurons below a minimum activity rate (fraction of bins with non-zero activity) threshold specified
        by the ``min_ar`` parameter will be excluded from the sampled data.

        Returns
        -------
        rasters_A, rasters_B - dictionaries with resampled data for all conditions from different trials
        """
        if ndata is None:
            ndata = self._max_conditioned_data
        if z_score is None:
            z_score = self._zscore

        resampled_rasters_A = {}
        resampled_rasters_B = {}

        for key in self.conditioned_rasters:
            if condition_names:
                condition_key = self._semantic_vectors[key]
            else:
                condition_key = key
            training, testing = sample_training_testing_from_rasters(self.conditioned_rasters[key],
                                                                     ndata=ndata,
                                                                     training_fraction=fraction,
                                                                     trials=self.conditioned_trial_index[key])
            resampled_rasters_A[condition_key] = training
            resampled_rasters_B[condition_key] = testing

        if z_score:
            for i in range(self.n_neurons):
                big_x = np.hstack(
                    [r[:, i] for r in resampled_rasters_A.values()] + [r[:, i] for r in resampled_rasters_B.values()])
                bigmean = np.nanmean(big_x)
                bigstd = np.nanstd(big_x)
                for key in resampled_rasters_A:
                    resampled_rasters_A[key][:, i] = (resampled_rasters_A[key][:, i] - bigmean) / bigstd
                    resampled_rasters_B[key][:, i] = (resampled_rasters_B[key][:, i] - bigmean) / bigstd

        if min_ar:
            X = np.vstack(
                [resampled_rasters_A[key] for key in resampled_rasters_A] + [resampled_rasters_B[key] for key in
                                                                             resampled_rasters_B])
            activityrate = np.nanmean(X > 0, 0)
            for key in resampled_rasters_A:
                resampled_rasters_A[key] = resampled_rasters_A[key][:, activityrate > min_ar]
                resampled_rasters_B[key] = resampled_rasters_B[key][:, activityrate > min_ar]

        return resampled_rasters_A, resampled_rasters_B

    # Dichotomy analysis functions

    def decode_dichotomy(self,
                         dichotomy: Union[str, list],
                         training_fraction: float,
                         cross_validations: int = 10,
                         ndata: Optional[int] = None,
                         shuffled: bool = False,
                         parallel: bool = False,
                         testing_trials: Optional[list] = None,
                         dic_key: Optional[str] = None,
                         subsample: Optional[float] = 0,
                         **kwargs) -> ndarray:
        """
        Function that performs cross-validated decoding of a specific dichotomy.
        Decoding is performed by sampling a balanced amount of data points from each condition in each class of the
        dichotomy, so to ensure that only the desired variable is analyzed by balancing confounds.
        Before sampling, each condition is individually divided into training and testing bins
        by using the ``self.trial`` array specified in the data structure when constructing the ``Decodanda`` object.


        Parameters
        ----------
            dichotomy : str || list
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
            dic_key:
                if specified, weights of the decoding analysis will be saved in self.decoding_weights using dic_key as the dictionary key.
            subsample:
                if >0, a random subsample of neurons of size=subsample will be used at each cross-validation

        Returns
        -------
            performances: list of decoding performance values for each cross-validation.

        Note
        ----
        ``dichotomy`` can be passed as a string or as a list.
        If a string is passed, it has to be a name of one of the variables specified in the conditions dictionary.

        If a list is passed, it needs to contain two lists in the shape [[...], [...]].
        Each sub list contains the conditions used to define one of the two decoded classes
        in binary notation.

        For example, if the data set has two variables
        ``stimulus`` :math:`\\in` {-1, 1} and ``action`` :math:`\\in` {-1, 1}, the condition
        ``stimulus=-1`` & ``action=-1`` will correspond to the binary notation ``'00'``,
        the condition ``stimulus=+1`` & ``action=-1`` will correspond to ``10`` and so on.
        Therefore, the notation:


        >>> dic = 'stimulus'

        is equivalent to

        >>> dic = [['00', '01'], ['10', '11']]

        and

        >>> dic = 'action'

        is equivalent to

        >>> dic = [['00', '10'], ['01', '11']]

        However, not all dichotomies have names (are semantic). For example, the dichotomy

        >>> [['01','10'], ['00', '11']]

        can only be defined using the binary notation.

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

        if type(dichotomy) == str:
            dic = self._dichotomy_from_key(dichotomy)
        else:
            dic = dichotomy
        if ndata is None and self.n_brains == 1:
            ndata = self._max_conditioned_data
        if ndata is None and self.n_brains > 1:
            ndata = max(self._max_conditioned_data, 2 * self.n_neurons)
        if subsample:
            self._generate_random_subset(subsample)
        if shuffled:
            self._shuffle_conditioned_arrays(dic)

        if self._verbose and not shuffled:
            print(dic, ndata)
            log_dichotomy(self, dic, ndata, 'Decoding')
            count = tqdm(range(cross_validations))
        else:
            count = range(cross_validations)

        if parallel:
            # TODO: add subsample to the parallel routine
            pool = Pool()
            res = pool.map(CrossValidator(classifier=self.classifier,
                                          conditioned_rasters=self.conditioned_rasters,
                                          conditioned_trial_index=self.conditioned_trial_index,
                                          dic=dic,
                                          training_fraction=training_fraction,
                                          ndata=ndata,
                                          subset=self.subset,
                                          semantic_vectors=self._semantic_vectors,
                                          z_score=self._zscore,
                                          dic_key=dic_key),
                           range(cross_validations))
            performances = np.asarray([r[0] for r in res])

            if len(res[0][1]):
                key = list(res[0][1].keys())[0]
                weights = {key: [r[1][key] for r in res]}

            print(performances, weights)

        else:
            performances = np.zeros(cross_validations)
            if self._verbose and not shuffled:
                print('\nLooping over decoding cross validation folds:')
            for i in count:
                if subsample:
                    self._generate_random_subset(subsample)

                performances[i] = self._one_cv_step(dic=dic, training_fraction=training_fraction, ndata=ndata,
                                                    shuffled=shuffled, testing_trials=testing_trials, dic_key=dic_key)
                if subsample:
                    self._generate_random_subset(self.n_neurons)
        if shuffled:
            self._order_conditioned_rasters()
        return np.asarray(performances)

    def CCGP_dichotomy(self, dichotomy: Union[str, list],
                       resamplings: int = 3,
                       ndata: Optional[int] = None,
                       max_semantic_dist: int = 1,
                       split_rule='OneOut',
                       shuffled: bool = False,
                       **kwargs):
        """
        Function that performs the cross-condition generalization performance analysis (CCGP, Bernardi et al. 2020, Cell)
        for a given variable, specified through its corresponding dichotomy. This function tests how well a given
        coding strategy for the given variable generalizes when the other variables are changed.


        Parameters
        ----------
            dichotomy : str || list
                The dichotomy corresponding to the variable to be tested, expressed in a double-list binary format, e.g. [['10', '11'], ['01', '00']], or as a variable name.
            resamplings:
                The number of iterations for each decoding analysis. The returned performance value is the average over these resamplings.
            ndata:
                The number of data points (population vectors) sampled for training and for testing for each condition.
            max_semantic_dist:
                The maximum semantic distance (number of variables that change value) between conditions in the held-out pair used to test the classifier.
            split_rule:
                The way conditions are split in training and testing. OneOut (default), name of a variable, or dichotomy in the double-list binary format. If OneOut is used, one pair of conditions is held out and the rest is used to train the classifier; if a variable is specified, then CCGP is computed specifically across that variable, balancing any third (or further) variables during sampling.
            shuffled:
                If True, the data is sampled according to geometrical null model for CCGP that keeps variables decodable but breaks the generalization. See Bernardi et al 2020 & Boyle, Posani et al. 2023.

        Returns
        -------
            performances: list of performance values for each cross-condition training-testing split.

        Note
        ----

        This function trains the ``self._classifier`` to decode the given variable in a sub-set
        of conditions, and tests it on the held-out set.

        The split of training and testing conditions is decided by the ``max_semantic_dist`` parameter: if set to 1,
        only pairs of conditions that have all variables in common except the specified one are held out to test the
        classifier.


            For example, if the data set has two variables
        ``stimulus`` :math:`\\in` {-1, 1} and ``action`` :math:`\\in` {-1, 1}, to compute CCGP for ``stimulus``
        with ``max_semantic_dist=1`` this function will train the classifier on

        ``(stimulus = -1, action = -1)`` vs. ``(stimulus = 1, action = -1)``

        And test it on

        ``(stimulus = -1, action = 1)`` vs. ``(stimulus = 1, action = 1)``

        note that action is kept fixed within the training and testing conditions.

        If instead we use ``max_semantic_dist=2``, all possible combinations are used, including training on

        ``(stimulus = -1, action = -1)`` vs. ``(stimulus = 1, action = 1)``

        and testing on

        ``(stimulus = -1, action = 1)`` vs. ``(stimulus = 1, action = -1)``


                ``dichotomy`` can be passed as a string or as a list.
        If a string is passed, it has to be a name of one of the variables specified in the conditions dictionary.

        If a list is passed, it needs to contain two lists in the shape [[...], [...]].
        Each sub list contains the conditions used to define one of the two decoded classes
        in binary notation.

        For example, if the data set has two variables
        ``stimulus`` :math:`\\in` {-1, 1} and ``action`` :math:`\\in` {-1, 1}, the condition
        ``stimulus=-1`` & ``action=-1`` will correspond to the binary notation ``'00'``,
        the condition ``stimulus=+1`` & ``action=-1`` will correspond to ``10`` and so on.

        Therefore, if ``stimulus`` is the first variable in the conditions dictionary, its corresponding dichotomy is

        >>> stimulus = [['00', '01'], ['10', '11']]

        Example
        -------
        >>> data = generate_synthetic_data(keyA='stimulus', keyB='action')
        >>> dec = Decodanda(data=data, conditions={'stimulus': [-1, 1], 'action': [-1, 1]})
        >>> perfs = dec.CCGP_dichotomy('stimulus')
        >>> perfs
        [0.82, 0.87] # 2 values

        """

        if type(dichotomy) == str:
            dic = self._dichotomy_from_key(dichotomy)
        else:
            dic = dichotomy

        if ndata is None and self.n_brains == 1:
            ndata = self._max_conditioned_data
        if ndata is None and self.n_brains > 1:
            ndata = max(self._max_conditioned_data, 2 * self.n_neurons)

        all_performances = []

        if not shuffled and self._verbose:
            log_dichotomy(self, dic, ndata, 'Cross-condition decoding')
            iterable = tqdm(range(resamplings))
        elif not shuffled:
            iterable = range(resamplings)
        else:
            iterable = range(1)

        for n in iterable:
            performances = []

            set_A = dic[0]
            set_B = dic[1]

            if split_rule == 'OneOut':
                # loop over all possible held-out pairs
                for i in range(len(set_A)):
                    for j in range(len(set_B)):
                        test_condition_A = set_A[i]
                        test_condition_B = set_B[j]

                        if hamming(string_bool(test_condition_A), string_bool(test_condition_B)) <= max_semantic_dist:
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

                            testing_array_A = sample_from_rasters(self.conditioned_rasters[test_condition_A],
                                                                  ndata=ndata)
                            testing_array_B = sample_from_rasters(self.conditioned_rasters[test_condition_B],
                                                                  ndata=ndata)

                            if shuffled:
                                rotation_A = np.arange(testing_array_A.shape[1]).astype(int)
                                rotation_B = np.arange(testing_array_B.shape[1]).astype(int)
                                np.random.shuffle(rotation_A)
                                np.random.shuffle(rotation_B)
                                testing_array_A = testing_array_A[:, rotation_A]
                                testing_array_B = testing_array_B[:, rotation_A]

                            if self._zscore:
                                big_raster = np.vstack(
                                    [training_array_A, training_array_B])  # z-scoring using the training data
                                big_mean = np.nanmean(big_raster, 0)
                                big_std = np.nanstd(big_raster, 0)
                                big_std[big_std == 0] = np.inf
                                training_array_A = (training_array_A - big_mean) / big_std
                                training_array_B = (training_array_B - big_mean) / big_std
                                testing_array_A = (testing_array_A - big_mean) / big_std
                                testing_array_B = (testing_array_B - big_mean) / big_std

                            self._train(training_array_A, training_array_B, label_A, label_B)
                            performance = self._test(testing_array_A, testing_array_B, label_A, label_B)
                            performances.append(performance)

            elif type(split_rule) == str:
                split_dichotomy = self._dichotomy_from_key(split_rule)
                training_conditions_A = [c for c in dic[0] if c in split_dichotomy[0]]
                training_conditions_B = [c for c in dic[1] if c in split_dichotomy[0]]

                testing_conditions_A = [c for c in dic[0] if c in split_dichotomy[1]]
                testing_conditions_B = [c for c in dic[1] if c in split_dichotomy[1]]

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

                if self._verbose:
                    print(f'\nCCGP: Training on {label_A} vs {label_B}')

                testing_array_A = []
                testing_array_B = []
                label_A_test = ''
                label_B_test = ''

                for ck in testing_conditions_A:
                    arr = sample_from_rasters(self.conditioned_rasters[ck], ndata=ndata)
                    testing_array_A.append(arr)
                    label_A_test += (self._semantic_vectors[ck] + ' ')

                for ck in testing_conditions_B:
                    arr = sample_from_rasters(self.conditioned_rasters[ck], ndata=ndata)
                    testing_array_B.append(arr)
                    label_B_test += (self._semantic_vectors[ck] + ' ')

                testing_array_A = np.vstack(testing_array_A)
                testing_array_B = np.vstack(testing_array_B)

                if self._verbose:
                    print(f'CCGP: Testing on {label_A_test} vs {label_B_test}')

                if shuffled:
                    rotation_A = np.arange(testing_array_A.shape[1]).astype(int)
                    rotation_B = np.arange(testing_array_B.shape[1]).astype(int)
                    np.random.shuffle(rotation_A)
                    np.random.shuffle(rotation_B)
                    testing_array_A = testing_array_A[:, rotation_A]
                    testing_array_B = testing_array_B[:, rotation_A]

                if self._zscore:
                    big_raster = np.vstack(
                        [training_array_A, training_array_B])  # z-scoring using the training data
                    big_mean = np.nanmean(big_raster, 0)
                    big_std = np.nanstd(big_raster, 0)
                    big_std[big_std == 0] = np.inf
                    training_array_A = (training_array_A - big_mean) / big_std
                    training_array_B = (training_array_B - big_mean) / big_std
                    testing_array_A = (testing_array_A - big_mean) / big_std
                    testing_array_B = (testing_array_B - big_mean) / big_std

                self._train(training_array_A, training_array_B, label_A, label_B)
                performance1 = self._test(testing_array_A, testing_array_B, label_A, label_B)

                self._train(testing_array_A, testing_array_B, label_A, label_B)
                performance2 = self._test(training_array_A, training_array_B, label_A, label_B)

                performances = [performance1, performance2]

            all_performances.append(performances)

        return np.nanmean(all_performances, 0)

    def parallelism_score_dichotomy(self, dichotomy: Union[str, list],
                                    max_semantic_dist: int = 1,
                                    shuffled: bool = False,
                                    method: str = 'pearson',
                                    return_combinations: bool = False):
        if type(dichotomy) == str:
            dic = self._dichotomy_from_key(dichotomy)
        else:
            dic = dichotomy

        ndata = 2 * self._max_conditioned_data

        coding_directions = []

        set_A = dic[0]
        set_B = dic[1]

        for i in range(len(set_A)):
            for j in range(len(set_B)):
                test_condition_A = set_A[i]
                test_condition_B = set_B[j]
                if hamming(string_bool(test_condition_A), string_bool(test_condition_B)) <= max_semantic_dist:
                    testing_array_A = sample_from_rasters(self.conditioned_rasters[test_condition_A], ndata=ndata)
                    testing_array_B = sample_from_rasters(self.conditioned_rasters[test_condition_B], ndata=ndata)

                    if shuffled:
                        rotation_A = np.arange(testing_array_A.shape[1]).astype(int)
                        rotation_B = np.arange(testing_array_B.shape[1]).astype(int)
                        np.random.shuffle(rotation_A)
                        np.random.shuffle(rotation_B)
                        testing_array_A = testing_array_A[:, rotation_A]
                        testing_array_B = testing_array_B[:, rotation_A]

                    if self._zscore:
                        big_raster = np.vstack([testing_array_A, testing_array_B])
                        big_mean = np.nanmean(big_raster, 0)
                        big_std = np.nanstd(big_raster, 0)
                        big_std[big_std == 0] = np.inf
                        testing_array_A = (testing_array_A - big_mean) / big_std
                        testing_array_B = (testing_array_B - big_mean) / big_std

                    vA = np.nanmean(testing_array_A, 0)
                    vB = np.nanmean(testing_array_B, 0)
                    coding_directions.append(vB - vA)

        parallelism_scores = []
        for i in range(len(coding_directions)):
            for j in range(i + 1, len(coding_directions)):
                if method == 'pearson':
                    parallelism_scores.append(scipy.stats.pearsonr(coding_directions[i], coding_directions[j])[0])
                elif method == 'cosine':
                    parallelism_scores.append(cosine(coding_directions[i], coding_directions[j]))
                elif method == 'spearman':
                    parallelism_scores.append(scipy.stats.spearmanr(coding_directions[i], coding_directions[j])[0])
                else:
                    raise ValueError(
                        "The specified method is not supported, please use one of: pearson, cosine, spearman")
        if return_combinations:
            return np.asarray(parallelism_scores)
        else:
            return np.nanmean(parallelism_scores)

    # Dichotomy analysis functions with null model

    def decode_with_nullmodel(self, dichotomy: Union[str, list],
                              training_fraction: float,
                              cross_validations: int = 10,
                              nshuffles: int = 10,
                              ndata: Optional[int] = None,
                              parallel: bool = False,
                              return_CV: bool = False,
                              testing_trials: Optional[list] = None,
                              plot: bool = False,
                              dic_key: Optional[str] = None,
                              subsample: Optional[int] = 0,
                              **kwargs) -> Tuple[Union[list, ndarray], ndarray]:
        """
        Function that performs cross-validated decoding of a specific dichotomy and compares the resulting values with
        a null model where the relationship between the neural data and the two sides of the dichotomy is
        shuffled.

        Decoding is performed by sampling a balanced amount of data points from each condition in each class of the
        dichotomy, so to ensure that only the desired variable is analyzed by balancing confounds.

        Before sampling, each condition is individually divided into training and testing bins
        by using the ``self.trial`` array specified in the data structure when constructing the ``Decodanda`` object.


        Parameters
        ----------
            dichotomy : str || list
                The dichotomy to be decoded, expressed in a double-list binary format, e.g. [['10', '11'], ['01', '00']], or as a variable name.
            training_fraction:
                the fraction of trials used for training in each cross-validation fold.
            cross_validations:
                the number of cross-validations.
            nshuffles:
                the number of null-model iterations of the decoding procedure.
            ndata:
                the number of data points (population vectors) sampled for training and for testing for each condition.
            parallel:
                if True, each cross-validation is performed by a dedicated thread (experimental, use with caution).
            return_CV:
                if True, invidual cross-validation values are returned in a list. Otherwise, the average performance over the cross-validation folds is returned.
            testing_trials:
                if specified, data sampled from the specified trial numbers will be used for testing, and the remaining ones for training.
            plot:
                if True, a visualization of the decoding results is shown.
            dic_key:
                if specified, weights of the decoding analysis will be saved in self.decoding_weights using dic_key as the dictionary key.
            subsample:
                if >0, a random subsample of neurons of size=subsample will be used at each cross-validation


        Returns
        -------
            performances, null_performances: list of decoding performance values for each cross-validation.


        See Also
        --------
        Decodanda.decode_dichotomy : The method used for each decoding iteration.


        Note
        ----
        ``dichotomy`` can be passed as a string or as a list.
        If a string is passed, it has to be a name of one of the variables specified in the conditions dictionary.

        If a list is passed, it needs to contain two lists in the shape [[...], [...]].
        Each sub list contains the conditions used to define one of the two decoded classes
        in binary notation.

        For example, if the data set has two variables
        ``stimulus`` :math:`\\in` {-1, 1} and ``action`` :math:`\\in` {-1, 1}, the condition
        ``stimulus=-1`` & ``action=-1`` will correspond to the binary notation ``'00'``,
        the condition ``stimulus=+1`` & ``action=-1`` will correspond to ``10`` and so on.
        Therefore, the notation:


        >>> dic = 'stimulus'

        is equivalent to

        >>> dic = [['00', '01'], ['10', '11']]

        and

        >>> dic = 'action'

        is equivalent to

        >>> dic = [['00', '10'], ['01', '11']]

        However, not all dichotomies have names (are semantic). For example, the dichotomy

        >>> [['01','10'], ['00', '11']]

        can only be defined using the binary notation.

        Note that this function gives you the flexibility to use sub-sets of conditions, for example

        >>> dic = [['10'], ['01']]

        will decode stimulus=1 & action=-1  vs.  stimulus=-1 & action=1


        Example
        -------
        >>> data = generate_synthetic_data(keyA='stimulus', keyB='action')
        >>> dec = Decodanda(data=data, conditions={'stimulus': [-1, 1], 'action': [-1, 1]})
        >>> perf, null = dec.decode_with_nullmodel('stimulus', training_fraction=0.75, cross_validations=10, nshuffles=20)
        >>> perf
        0.88
        >>> null
        [0.51, 0.54, 0.48, ..., 0.46] # 25 values
        """

        if type(dichotomy) == str:
            dic = self._dichotomy_from_key(dichotomy)
        else:
            dic = dichotomy

        d_performances = self.decode_dichotomy(dichotomy=dic,
                                               training_fraction=training_fraction,
                                               cross_validations=cross_validations,
                                               ndata=ndata,
                                               parallel=parallel,
                                               testing_trials=testing_trials,
                                               dic_key=dic_key,
                                               subsample=subsample)
        if return_CV:
            data_performance = d_performances
        else:
            data_performance = np.nanmean(d_performances)

        if self._verbose and nshuffles:
            print(
                "\n[decode_with_nullmodel]\t data <p> = %.2f" % np.nanmean(d_performances))
            print('\n[decode_with_nullmodel]\tLooping over null model shuffles.')
            count = tqdm(range(nshuffles))
        else:
            count = range(nshuffles)

        null_model_performances = np.zeros(nshuffles)

        for n in count:
            performances = self.decode_dichotomy(dichotomy=dic,
                                                 training_fraction=training_fraction,
                                                 cross_validations=cross_validations,
                                                 ndata=ndata,
                                                 parallel=parallel,
                                                 testing_trials=testing_trials,
                                                 shuffled=True,
                                                 dic_key=dic_key,
                                                 subsample=subsample)

            null_model_performances[n] = np.nanmean(performances)

        if plot:
            visualize_decoding(self, dic, d_performances, null_model_performances,
                               training_fraction=training_fraction, ndata=ndata, testing_trials=testing_trials)

        return data_performance, null_model_performances

    def CCGP_with_nullmodel(self, dichotomy: Union[str, list],
                            resamplings: int = 5,
                            nshuffles: int = 25,
                            ndata: Optional[int] = None,
                            max_semantic_dist: int = 1,
                            split_rule='OneOut',
                            return_combinations: bool = False,
                            **kwargs):

        """
                Function that performs the cross-condition generalization performance analysis (CCGP, Bernardi et al. 2020, Cell)
                for a given variable, specified through its corresponding dichotomy.

                    This function tests how well a given coding strategy for the given variable generalizes
                when the other variables are changed and compares the
                resulting values with a geometrical null model that keeps variables decodable but randomly
                displaces conditions in the neural activity space, hence breaking any coding parallelism and generizability.
                See Bernardi et al 2020 & Boyle, Posani et al. 2023 for more details.

                Parameters
                ----------
                    dichotomy : str || list
                        The dichotomy corresponding to the variable to be tested, expressed in a double-list binary format, e.g. [['10', '11'], ['01', '00']], or as a variable name.
                    resamplings:
                        The number of iterations for each decoding analysis. The returned performance value is the average over these resamplings.
                    nshuffles:
                        The number of null-model iterations for the CCGP analysis.
                    ndata:
                        The number of data points (population vectors) sampled for training and for testing for each condition.
                    max_semantic_dist:
                        The maximum semantic distance (number of variables that change value) between conditions in the held-out pair used to test the classifier.
                    split_rule:
                        The way conditions are split in training and testing. OneOut (default), name of a variable, or dichotomy in the double-list binary format. If OneOut is used, one pair of conditions is held out and the rest is used to train the classifier; if a variable is specified, then CCGP is computed specifically across that variable, balancing any third (or further) variables during sampling.
                    return_combinations:
                        If True, returns all the individual performances for cross-conditions train-test splits, otherwise returns the average over combinations.


                Returns
                -------
                    ccgp: mean of performance values for each cross-condition training-testing split (or list, if ``return_combinations=True``).
                    null: a list of null values for the mean ccgp

                Note
                ----

                This function trains the ``self._classifier`` to decode the given variable in a sub-set
                of conditions, and tests it on the held-out set.

                The split of training and testing conditions is decided by the ``max_semantic_dist`` parameter: if set to 1,
                only pairs of conditions that have all variables in common except the specified one are held out to test the
                classifier.


                    For example, if the data set has two variables
                ``stimulus`` :math:`\\in` {-1, 1} and ``action`` :math:`\\in` {-1, 1}, to compute CCGP for ``stimulus``
                with ``max_semantic_dist=1`` this function will train the classifier on

                ``(stimulus = -1, action = -1)`` vs. ``(stimulus = 1, action = -1)``

                And test it on

                ``(stimulus = -1, action = 1)`` vs. ``(stimulus = 1, action = 1)``

                note that action is kept fixed within the training and testing conditions.

                If instead we use ``max_semantic_dist=2``, all possible combinations are used, including training on

                ``(stimulus = -1, action = -1)`` vs. ``(stimulus = 1, action = 1)``

                and testing on

                ``(stimulus = -1, action = 1)`` vs. ``(stimulus = 1, action = -1)``


                        ``dichotomy`` can be passed as a string or as a list.
                If a string is passed, it has to be a name of one of the variables specified in the conditions dictionary.

                If a list is passed, it needs to contain two lists in the shape [[...], [...]].
                Each sub list contains the conditions used to define one of the two decoded classes
                in binary notation.

                For example, if the data set has two variables
                ``stimulus`` :math:`\\in` {-1, 1} and ``action`` :math:`\\in` {-1, 1}, the condition
                ``stimulus=-1`` & ``action=-1`` will correspond to the binary notation ``'00'``,
                the condition ``stimulus=+1`` & ``action=-1`` will correspond to ``10`` and so on.

                Therefore, if ``stimulus`` is the first variable in the conditions dictionary, its corresponding dichotomy is

                >>> stimulus = [['00', '01'], ['10', '11']]

                Example
                -------
                >>> data = generate_synthetic_data(keyA='stimulus', keyB='action')
                >>> dec = Decodanda(data=data, conditions={'stimulus': [-1, 1], 'action': [-1, 1]})
                >>> perf, null = dec.CCGP_with_nullmodel('stimulus', nshuffles=10)
                >>> perf
                0.85
                >>> null
                [0.44, 0.48, ..., 0.54] # 10 values
                """

        performances = self.CCGP_dichotomy(dichotomy=dichotomy, resamplings=resamplings, ndata=ndata,
                                           max_semantic_dist=max_semantic_dist, split_rule=split_rule)

        if return_combinations:
            ccgp = performances
        else:
            ccgp = np.nanmean(performances)

        if self._verbose and nshuffles:
            print("\t\t[CCGP_with_nullmodel]\t\t----- Data: <p> = %.2f -----\n" % np.nanmean(performances))
            count = tqdm(range(nshuffles))
        else:
            count = range(nshuffles)

        shuffled_ccgp = []
        for n in count:
            performances = self.CCGP_dichotomy(dichotomy=dichotomy,
                                               resamplings=resamplings,
                                               ndata=ndata,
                                               max_semantic_dist=max_semantic_dist,
                                               split_rule=split_rule,
                                               shuffled=True)
            if return_combinations:
                shuffled_ccgp.append(performances)
            else:
                shuffled_ccgp.append(np.nanmean(performances))

        return ccgp, shuffled_ccgp

    def PS_with_nullmodel(self, dichotomy: Union[str, list],
                          nshuffles: int = 25,
                          max_semantic_dist: int = 1,
                          method: str = 'pearson',
                          return_combinations: bool = False,
                          **kwargs):

        scores = self.parallelism_score_dichotomy(dichotomy=dichotomy,
                                                  method=method,
                                                  max_semantic_dist=max_semantic_dist,
                                                  return_combinations=return_combinations)

        if self._verbose and nshuffles:
            print("\t\t[PS_with_nullmodel]\t\t----- Data: <p> = %.2f -----\n" % np.nanmean(scores))
            count = tqdm(range(nshuffles))
        else:
            count = range(nshuffles)

        shuffled_scores = []
        for n in count:
            scores_null = self.parallelism_score_dichotomy(dichotomy=dichotomy,
                                                           method=method,
                                                           max_semantic_dist=max_semantic_dist,
                                                           return_combinations=return_combinations,
                                                           shuffled=True)
            shuffled_scores.append(scores_null)

        return scores, shuffled_scores

    # Decoding analysis for semantic dichotomies

    def decode(self, training_fraction: float,
               cross_validations: int = 10,
               nshuffles: int = 10,
               ndata: Optional[int] = None,
               subsample: Optional[int] = 0,
               parallel: bool = False,
               non_semantic: bool = False,
               return_CV: bool = False,
               testing_trials: Optional[list] = None,
               plot: bool = False,
               ax: Optional[plt.Axes] = None,
               plot_all: bool = False,
               **kwargs):

        """
        Main function to decode the variables specified in the ``conditions`` dictionary.

        It returns a single decoding value per variable which represents the average over
        the cross-validation folds.

        It also returns an array of null-model values for each variable to test the significance of
        the corresponding decoding result.

        Notes
        -----

        Each decoding analysis is performed by first re-sampling an equal number of data points
        from each condition (combination of variable values), so to ensure that possible confounds
        due to correlated conditions are balanced out.


        Before sampling, each condition is individually divided into training and testing bins
        by using the ``self.trial`` array specified in the data structure when constructing the ``Decodanda`` object.


        To generate the null model values, the relationship between the neural data and
        the decoded variable is randomly shuffled. Eeach null model value corresponds to the
        average across ``cross_validations``` iterations after a single data shuffle.


        If ``non_semantic=True``, dichotomies that do not correspond to variables will also be decoded.
        Note that, in the case of 2 variables, there is only one non-semantic dichotomy
        (corresponding to grouping together conditions that have the same XOR value in the
        binary notation: ``[['10', '01'], ['11', '00']]``). However, the number of non-semantic dichotomies
        grows exponentially with the number of conditions, so use with caution if more than two variables
        are specified in the conditions dictionary.


        Parameters
        ----------
        training_fraction:
            the fraction of trials used for training in each cross-validation fold.
        cross_validations:
            the number of cross-validations.
        nshuffles:
            the number of null-model iterations of the decoding procedure.
        ndata:
            the number of data points (population vectors) sampled for training and for testing for each condition.
        subsample:
            if >0, a random subsample of neurons of size=subsample will be used at each cross-validation
        parallel:
            if True, each cross-validation is performed by a dedicated thread (experimental, use with caution).
        return_CV:
            if True, invidual cross-validation values are returned in a list. Otherwise, the average performance over the cross-validation folds is returned.
        testing_trials:
            if specified, data sampled from the specified trial numbers will be used for testing, and the remaining ones for training.
        non_semantic:
            if True, non-semantic dichotomies (i.e., dichotomies that do not correspond to a variable) will also be decoded.
        plot:
            if True, a visualization of the decoding results is shown.
        ax:
            if specified and ``plot=True``, the results will be displayed in the specified axis instead of a new figure.
        plot_all:
            if True, a more in-depth visualization of the decoding results and of the decoded data is shown.


        Returns
        -------
            perfs:
                a dictionary containing the decoding performances for all variables in the form of ``{var_name_1: performance1, var_name_2: performance2, ...}``
            null:
                a dictionary containing an array of null model decoding performance for each variable in the form ``{var_name_1: [...], var_name_2: [...], ...}``.

        See Also
        --------
            Decodanda.decode_with_nullmodel: The method used for each decoding analysis.


        Example
        -------
        >>> from decodanda import Decodanda, generate_synthetic_data
        >>> data = generate_synthetic_data(keyA='stimulus', keyB='action')
        >>> dec = Decodanda(data=data, conditions={'stimulus': [-1, 1], 'action': [-1, 1]})
        >>> perfs, null = dec.decode(training_fraction=0.75, cross_validations=10, nshuffles=20)
        >>> perfs
        {'stimulus': 0.88, 'action': 0.85}  # mean over 10 cross-validation folds
        >>> null
        {'stimulus': [0.51, ..., 0.46], 'action': [0.48, ..., 0.55]}  # null model means, 20 values each
        """

        semantic_dics, semantic_keys = self._find_semantic_dichotomies()

        perfs = {}
        perfs_nullmodel = {}
        for key, dic in zip(semantic_keys, semantic_dics):
            if self._verbose:
                print("\nTesting decoding performance for semantic dichotomy: ", key)
            performance, null_model_performances = self.decode_with_nullmodel(
                dic,
                training_fraction,
                cross_validations=cross_validations,
                ndata=ndata,
                nshuffles=nshuffles,
                parallel=parallel,
                return_CV=return_CV,
                testing_trials=testing_trials,
                plot=plot_all,
                subsample=subsample)

            perfs[key] = performance
            perfs_nullmodel[key] = null_model_performances

        if non_semantic and len(self.conditions) == 2:
            xor_dic = [['01', '10'], ['00', '11']]
            perfs_xor, perfs_null_xor = self.decode_with_nullmodel(dichotomy=xor_dic,
                                                                   training_fraction=training_fraction,
                                                                   cross_validations=cross_validations,
                                                                   nshuffles=nshuffles,
                                                                   parallel=parallel,
                                                                   ndata=ndata,
                                                                   return_CV=return_CV,
                                                                   testing_trials=testing_trials,
                                                                   plot=plot_all,
                                                                   subsample=subsample)
            perfs['XOR'] = perfs_xor
            perfs_nullmodel['XOR'] = perfs_null_xor

        if non_semantic and len(self.conditions) > 2:
            dics = self._find_nonsemantic_dichotomies()
            for dic in dics:
                dic_key = '_'.join(dic[0]) + '__' + '_'.join(dic[1])
                perfs_dic, null_dic = self.decode_with_nullmodel(dichotomy=dic,
                                                                 training_fraction=training_fraction,
                                                                 cross_validations=cross_validations,
                                                                 nshuffles=nshuffles,
                                                                 parallel=parallel,
                                                                 ndata=ndata,
                                                                 return_CV=return_CV,
                                                                 testing_trials=testing_trials,
                                                                 plot=plot_all,
                                                                 subsample=subsample)
                perfs[dic_key] = perfs_dic
                perfs_nullmodel[dic_key] = null_dic

        if plot:
            if not ax:
                f, ax = plt.subplots(figsize=(0.5 + 1.8 * len(perfs.keys()), 3.5))
            plot_perfs_null_model(perfs, perfs_nullmodel, ylabel='Decoding performance', ax=ax, marker='o', **kwargs)

        return perfs, perfs_nullmodel

    # Geometrical analysis for semantic dichotomies

    def CCGP(self, resamplings=5,
             nshuffles: int = 25,
             ndata: Optional[int] = None,
             max_semantic_dist: int = 1,
             plot: bool = False,
             ax: Optional[plt.Axes] = None,
             **kwargs):

        """
        Main function that performs the cross-condition generalization performance analysis (CCGP, Bernardi et al. 2020, Cell)
        for the variables specified through the ``conditions`` dictionary.

        It returns a single ccgp value per variable which represents the average over
        all cross-condition train-test splits. This function uses split_rule='OneOut' as a default.

        It also returns an array of null-model values for each variable to test the significance of
        the corresponding ccgp result. The employed geometrical null model keeps variables decodable but randomly
        displaces conditions in the neural activity space, hence breaking any coding parallelism and generizability.
        See Bernardi et al 2020 & Boyle, Posani et al. 2023 for more details.

        Parameters
        ----
            resamplings:
                The number of iterations for each decoding analysis. The returned performance value is the average over these resamplings.
            nshuffles:
                The number of null-model iterations for the CCGP analysis.
            ndata:
                The number of data points (population vectors) sampled for training and for testing for each condition.
            max_semantic_dist:
                The maximum semantic distance (number of variables that change value) between conditions in the held-out pair used to test the classifier.
            plot:
                if True, a visualization of the decoding results is shown.
            ax:
                if specified and ``plot=True``, the results will be displayed in the specified axis instead of a new figure.

        Returns
        -------
            performance: mean of performance values for each cross-condition training-testing split.
            null: a list of null values for the generalization performance

        See Also
        --------
        Decodanda.CCGP_with_nullmodel

        Note
        ----

        For each variable, this function trains the ``self._classifier`` to decode the given variable in a sub-set
        of conditions, and tests it on the held-out set.

        The split of training and testing conditions is performed by keeping the semantic distance between held out
        conditions to 1 (``max_semantic_dist=1`` in the CCGP_dichotomy function).

        For example, if the data set has two variables:

        ``stimulus`` :math:`\\in` {-1, 1} and ``action`` :math:`\\in` {-1, 1}, to compute CCGP for ``stimulus``

        This function will train the classifier on

        ``(stimulus = -1, action = -1)`` vs. ``(stimulus = 1, action = -1)``

        And test it on

        ``(stimulus = -1, action = 1)`` vs. ``(stimulus = 1, action = 1)``

        And vice-versa. Note that action is kept fixed within the training and testing conditions.

        Example
        -------
        >>> data = generate_synthetic_data(keyA='stimulus', keyB='action')
        >>> dec = Decodanda(data=data, conditions={'stimulus': [-1, 1], 'action': [-1, 1]})
        >>> perfs, null = dec.CCGP(nshuffles=10)
        >>> perfs
        {'stimulus': 0.81, 'action': 0.79}  # each value is the mean over 2 cross-condition train-test splits
        >>> null
        {'stimulus': [0.51, ..., 0.46], 'action': [0.48, ..., 0.55]}  # null model means, 10 values each
        """

        semantic_dics, semantic_keys = self._find_semantic_dichotomies()

        ccgp = {}
        ccgp_nullmodel = {}
        for key, dic in zip(semantic_keys, semantic_dics):
            if self._verbose:
                print("\nTesting CCGP for semantic dichotomy: ", key)
            data_ccgp, null_ccgps = self.CCGP_with_nullmodel(dichotomy=dic,
                                                             resamplings=resamplings,
                                                             nshuffles=nshuffles,
                                                             ndata=ndata,
                                                             max_semantic_dist=max_semantic_dist)
            ccgp[key] = data_ccgp
            ccgp_nullmodel[key] = null_ccgps

        if plot:
            if not ax:
                f, ax = plt.subplots(figsize=(0.5 + 1.8 * len(semantic_dics), 3.5))
            plot_perfs_null_model(ccgp, ccgp_nullmodel, ylabel='CCGP', ax=ax, marker='s', **kwargs)

        return ccgp, ccgp_nullmodel

    def PS(self, nshuffles: int = 25,
           max_semantic_dist: int = 1,
           method: str = 'pearson',
           plot: bool = False,
           ax: Optional[plt.Axes] = None,
           **kwargs):

        semantic_dics, semantic_keys = self._find_semantic_dichotomies()

        ps = {}
        ps_nullmodel = {}
        for key, dic in zip(semantic_keys, semantic_dics):
            if self._verbose:
                print("\nTesting PS for semantic dichotomy: ", key)
            data_ps, null_ps = self.PS_with_nullmodel(dichotomy=dic,
                                                      nshuffles=nshuffles,
                                                      method=method,
                                                      max_semantic_dist=max_semantic_dist)
            ps[key] = data_ps
            ps_nullmodel[key] = null_ps

        if plot:
            if not ax:
                f, ax = plt.subplots(figsize=(0.5 + 1.8 * len(semantic_dics), 3.5))
            plot_perfs_null_model(ps, ps_nullmodel, ylabel='Parallelism Score', ax=ax, ylow=-1.05, yhigh=1.05, chance=0,
                                  **kwargs)

        return ps, ps_nullmodel

    def semantic_score_geometry(self,
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
            # print(i, res)
            decoding_results.append(res)
            decoding_null.append(null)

        # CCGP all dichotomies
        CCGP_results = []
        CCGP_null = []
        for i, dic in enumerate(all_dics):
            # print(dic)
            res, null = self.CCGP_with_nullmodel(dic,
                                                 nshuffles=nshuffles,
                                                 ndata=ndata,
                                                 max_semantic_dist=self.n_conditions)
            # print(i, res)
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

        dichotomies_data = [all_dics, semantic_overlap]
        decoding_data = [decoding_results, decoding_null]
        CCGP_data = [CCGP_results, CCGP_null]

        return dichotomies_data, decoding_data, CCGP_data

    def shattering_dimensionality(self,
                                  training_fraction: float = 0.75,
                                  cross_validations: int = 10,
                                  nshuffles: int = 10,
                                  ndata: Optional[int] = None,
                                  subsample: Optional[int] = 0,
                                  p_threshold: float = 0.01,
                                  visualize: bool = True,
                                  semantic_names: Optional[dict] = None,
                                  **kwargs):
        """
        This function computes shattering dimensionality as defined in Bernardi et al. 2020, i.e., as
        the number of balanced dichotomies that a linear decoder can classify above chance levels.

        Parameters
        ----------
        training_fraction:
            the fraction of trials used for training in each cross-validation fold.
        cross_validations:
            the number of cross-validations.
        nshuffles:
            the number of null-model iterations of the decoding procedure.
        ndata:
            the number of data points (population vectors) sampled for training and for testing for each condition.
        subsample:
            if >0, a random subsample of neurons of size=subsample will be used at each cross-validation.
        p_threshold:
            p-value threshold (z-score from the null model) to consider a performance as statistically significant.
        visualize:
            if ``True``, the decoding results are shown in a figure.


        Returns
        -------
        shattering_dim:
            shattering dimensionality
        perfs:
            dictionary of decoding performance per dichotomy
        null:
            dictionary of lists of null model values per dichotomy
        """

        all_dics_names, all_dics = generate_dichotomies(self.n_conditions)
        semantic_overlap = []
        dic_name = []
        is_semantic = []
        perfs = {}
        nulls = {}

        for i, dic in enumerate(all_dics):
            semantic_overlap.append(semantic_score(dic))
            is_semantic.append(str(self._dic_key(dic)))
            dic_name.append(all_dics_names[i])
        semantic_overlap = np.asarray(semantic_overlap)

        # sorting dichotomies wrt semantic overlap
        dic_name = np.asarray(dic_name)[np.argsort(semantic_overlap)[::-1]]
        all_dics = list(np.asarray(all_dics)[np.argsort(semantic_overlap)[::-1]])
        is_semantic = list(np.asarray(is_semantic)[np.argsort(semantic_overlap)[::-1]])
        semantic_overlap = semantic_overlap[np.argsort(semantic_overlap)[::-1]]
        semantic_overlap = (semantic_overlap - np.min(semantic_overlap)) / (
                np.max(semantic_overlap) - np.min(semantic_overlap))

        # decoding all dichotomies
        for i, dic in tqdm(enumerate(all_dics)):
            res, null = self.decode_with_nullmodel(dic,
                                                   training_fraction=training_fraction,
                                                   cross_validations=cross_validations,
                                                   nshuffles=nshuffles,
                                                   ndata=ndata,
                                                   subsample=subsample)
            perfs[dic_name[i]] = res
            nulls[dic_name[i]] = null

        ps = np.asarray([z_pval(perfs[dic_name[i]], nulls[dic_name[i]])[1] for i in range(len(dic_name))])
        shattering_dim = np.nanmean(ps < p_threshold)

        # plotting
        if visualize:
            f, ax = plt.subplots(figsize=(6, 3))
            if self.n_conditions > 2:
                ax.set_xlabel('Dichotomy (ordered by semantic score)')
            else:
                ax.set_xlabel('Dichotomy')

            ax.set_ylabel('Decoding Performance')
            ax.axhline([0.5], color='k', linestyle='--', alpha=0.5)
            ax.set_xticks([])
            ax.set_ylim([0, 1.05])
            sns.despine(f)

            # visualize Decoding
            for i in range(len(all_dics)):
                if z_pval(perfs[dic_name[i]], nulls[dic_name[i]])[1] < 0.01:
                    ax.scatter(i, perfs[dic_name[i]], marker='o',
                               color=cm.cool(int(semantic_overlap[i] * 255)), edgecolor='k',
                               s=110, linewidth=2, linestyle='-')
                else:
                    ax.scatter(i, perfs[dic_name[i]], marker='o',
                               color=cm.cool(int(semantic_overlap[i] * 255)), edgecolor='none',
                               s=110, linewidth=0, linestyle='')
                ax.errorbar(i, np.nanmean(nulls[dic_name[i]]), np.nanstd(nulls[dic_name[i]]), color='k', alpha=0.3)
                if semantic_names is not None:
                    if dic_name[i] in list(semantic_names.keys()):
                        ax.text(i, max(0.61, perfs[dic_name[i]] + 0.06), semantic_names[dic_name[i]], rotation=90,
                                fontsize=6, color='k',
                                ha='center', fontweight='bold')

            return shattering_dim, perfs, nulls, f
        else:
            return shattering_dim, perfs, nulls

    def shattering_generalization(self,
                                  nshuffles: int = 10,
                                  ndata: Optional[int] = None,
                                  p_threshold: float = 0.01,
                                  visualize: bool = True,
                                  semantic_names: Optional[dict] = None,
                                  **kwargs):
        """
        [WIP] This function computes shattering generalization defined as the number of balanced dichotomies that
        have a above-chance CCGP.

        Parameters
        ----------
        nshuffles:
            the number of null-model iterations of the decoding procedure.
        ndata:
            the number of data points (population vectors) sampled for training and for testing for each condition.
        p_threshold:
            p-value threshold (z-score from the null model) to consider a performance as statistically significant.
        visualize:
            if ``True``, the decoding results are shown in a figure.


        Returns
        -------
        shattering_gen:
            shattering dimensionality
        perfs:
            dictionary of decoding performance per dichotomy
        null:
            dictionary of lists of null model values per dichotomy
        """

        all_dics_names, all_dics = generate_dichotomies(self.n_conditions)
        semantic_overlap = []
        dic_name = []
        is_semantic = []
        perfs = {}
        nulls = {}

        for i, dic in enumerate(all_dics):
            semantic_overlap.append(semantic_score(dic))
            is_semantic.append(str(self._dic_key(dic)))
            dic_name.append(all_dics_names[i])
        semantic_overlap = np.asarray(semantic_overlap)

        # sorting dichotomies wrt semantic overlap
        dic_name = np.asarray(dic_name)[np.argsort(semantic_overlap)[::-1]]
        all_dics = list(np.asarray(all_dics)[np.argsort(semantic_overlap)[::-1]])
        is_semantic = list(np.asarray(is_semantic)[np.argsort(semantic_overlap)[::-1]])
        semantic_overlap = semantic_overlap[np.argsort(semantic_overlap)[::-1]]
        semantic_overlap = (semantic_overlap - np.min(semantic_overlap)) / (
                np.max(semantic_overlap) - np.min(semantic_overlap))

        # CCGP for all dichotomies
        for i, dic in tqdm(enumerate(all_dics)):
            res, null = self.CCGP_with_nullmodel(dic, resamplings=2,
                                                 nshuffles=nshuffles, ndata=ndata,
                                                 max_semantic_dist=99)

            perfs[dic_name[i]] = res
            nulls[dic_name[i]] = null

        ps = np.asarray([z_pval(perfs[dic_name[i]], nulls[dic_name[i]])[1] for i in range(len(dic_name))])
        shattering_gen = np.nanmean(ps < p_threshold)

        # plotting
        if visualize:
            f, ax = plt.subplots(figsize=(6, 3))
            if self.n_conditions > 2:
                ax.set_xlabel('Dichotomy (ordered by semantic score)')
            else:
                ax.set_xlabel('Dichotomy')

            ax.set_ylabel('CCGP')
            ax.axhline([0.5], color='k', linestyle='--', alpha=0.5)
            ax.set_xticks([])
            ax.set_ylim([0, 1.05])
            sns.despine(f)

            # visualize Decoding
            for i in range(len(all_dics)):
                if z_pval(perfs[dic_name[i]], nulls[dic_name[i]])[1] < 0.01:
                    ax.scatter(i, perfs[dic_name[i]], marker='o',
                               color=cm.cool(int(semantic_overlap[i] * 255)), edgecolor='k',
                               s=110, linewidth=2, linestyle='-')
                else:
                    ax.scatter(i, perfs[dic_name[i]], marker='o',
                               color=cm.cool(int(semantic_overlap[i] * 255)), edgecolor='none',
                               s=110, linewidth=0, linestyle='')
                ax.errorbar(i, np.nanmean(nulls[dic_name[i]]), np.nanstd(nulls[dic_name[i]]), color='k', alpha=0.3)
                if semantic_names is not None:
                    if dic_name[i] in list(semantic_names.keys()):
                        ax.text(i, max(0.61, perfs[dic_name[i]] + 0.06), semantic_names[dic_name[i]], rotation=90,
                                fontsize=6, color='k',
                                ha='center', fontweight='bold')

            return shattering_gen, perfs, nulls, f
        else:
            return shattering_gen, perfs, nulls

    def CVI(self, training_fraction: float = 0.75,
            cross_validations: int = 10,
            nshuffles: int = 10,
            ndata: Optional[int] = None,
            return_splits: bool = False,
            signed=False
            ):
        if ndata is None:
            ndata = 2 * self._max_conditioned_data

        dics, vars = self._find_semantic_dichotomies()
        # data
        results = {}
        for v1 in range(len(vars)):
            var1 = vars[v1]
            dic1 = dics[v1]
            for v2 in range(len(vars)):
                if v2 != v1:
                    var2 = vars[v2]
                    dic2 = dics[v2]
                    results[f'{var1}-{var2}'] = []
                    for k in range(cross_validations):
                        perf = self._one_X_cv_step(dic1, dic2, training_fraction, ndata)
                        results[f'{var1}-{var2}'].append(perf)
                    if signed:
                        results[f'{var1}-{var2}'] = np.nanmean(np.asarray(results[f'{var1}-{var2}']))
                    else:
                        results[f'{var1}-{var2}'] = 0.5 + np.abs(
                            np.nanmean(np.asarray(results[f'{var1}-{var2}']) - 0.5))
        # null
        null = {key: [] for key in results}
        for n in range(nshuffles):
            self._shuffle_conditioned_arrays(dic='XOR')
            for v1 in range(len(vars)):
                var1 = vars[v1]
                dic1 = dics[v1]
                for v2 in range(len(vars)):
                    if v2 != v1:
                        var2 = vars[v2]
                        dic2 = dics[v2]
                        null_n = []
                        for k in range(cross_validations):
                            perf = self._one_X_cv_step(dic1, dic2, training_fraction, ndata)
                            null_n.append(perf)
                        null[f'{var1}-{var2}'].append(np.nanmean(null_n))
            self._order_conditioned_rasters()
        if not return_splits:
            megakey = '-'.join(vars)
            results_combined = {megakey: np.nanmean([results[key] for key in results])}
            null_combined = {megakey: [np.nanmean([null[key][i] for key in null]) for i in range(nshuffles)]}
            return results_combined, null_combined
        return results, null

    # Utilities

    def visualize_PCA(self, **kwargs):
        fig = visualize_PCA(self, **kwargs)
        return fig

    # __init__ utilities

    def _divide_data_into_conditions(self, sessions):
        # TODO: rename sessions into datasets?
        # TODO: make sure conditions don't overlap somehow

        for si, session in enumerate(sessions):

            if self._verbose:
                if hasattr(session, 'name'):
                    print("\t\t[Decodanda]\tbuilding conditioned rasters for session %s" % session.name)
                else:
                    print("\t\t[Decodanda]\tbuilding conditioned rasters for session %u" % si)

            session_conditioned_rasters = {}
            session_conditioned_trial_index = {}

            # exclude inactive neurons across the specified conditions
            array = getattr(session, self._neural_attr)
            total_mask = np.zeros(len(array)) > 0

            for condition_vec in self._condition_vectors:
                mask = np.ones(len(array)) > 0
                for i, sk in enumerate(self._semantic_keys):
                    semantic_values = list(self.conditions[sk])
                    mask_i = self.conditions[sk][semantic_values[condition_vec[i]]](session)
                    mask = mask & mask_i
                total_mask = total_mask | mask

            min_activity_mask = np.sum(array[total_mask] != 0, 0) >= self._min_activations_per_cell

            for condition_vec in self._condition_vectors:
                # get the array from the session object
                array = getattr(session, self._neural_attr)
                array = array[:, min_activity_mask]

                # create a mask that becomes more and more restrictive by iterating on semanting conditions
                mask = np.ones(len(array)) > 0
                for i, sk in enumerate(self._semantic_keys):
                    semantic_values = list(self.conditions[sk])
                    mask_i = self.conditions[sk][semantic_values[condition_vec[i]]](session)
                    mask = mask & mask_i

                # select bins conditioned on the semantic behavioural vector
                conditioned_raster = array[mask, :]

                # Define trial logic
                def condition_no(cond):
                    no = 0
                    for i in range(len(cond)):
                        no += cond[i] * 10 ** (i + 3)
                    return no

                if self._trial_attr is not None:
                    conditioned_trial = getattr(session, self._trial_attr)[mask]
                elif self._trial_chunk is None:
                    if self._verbose:
                        print('[Decodanda]\tUsing contiguous chunks of the same labels as trials.')
                    conditioned_trial = contiguous_chunking(mask)[mask]
                    conditioned_trial += condition_no(condition_vec)
                else:
                    conditioned_trial = contiguous_chunking(mask, self._trial_chunk)[mask]
                    conditioned_trial += condition_no(condition_vec)

                if self._exclude_contiguous_trials:
                    contiguous_chunks = contiguous_chunking(mask)[mask]
                    nc_mask = non_contiguous_mask(contiguous_chunks, conditioned_trial)
                    conditioned_raster = conditioned_raster[nc_mask, :]
                    conditioned_trial = conditioned_trial[nc_mask]

                # exclude empty time bins (only for binary discrete decoding)
                if self._exclude_silent:
                    active_mask = np.sum(conditioned_raster, 1) > 0
                    conditioned_raster = conditioned_raster[active_mask, :]
                    conditioned_trial = conditioned_trial[active_mask]

                # squeeze into trials
                if self._trial_average:
                    unique_trials = np.unique(conditioned_trial[~np.isnan(conditioned_trial)])
                    squeezed_raster = []
                    squeezed_trial_index = []
                    for t in unique_trials:
                        trial_raster = conditioned_raster[conditioned_trial == t]
                        squeezed_raster.append(np.nanmean(trial_raster, 0))
                        squeezed_trial_index.append(t)
                    # set the new arrays
                    conditioned_raster = np.asarray(squeezed_raster)
                    conditioned_trial = np.asarray(squeezed_trial_index)

                # set the conditioned neural data in the conditioned_rasters dictionary
                session_conditioned_rasters[string_bool(condition_vec)] = conditioned_raster
                session_conditioned_trial_index[string_bool(condition_vec)] = conditioned_trial

                if self._verbose:
                    semantic_vector_string = []
                    for i, sk in enumerate(self._semantic_keys):
                        semantic_values = list(self.conditions[sk])
                        semantic_vector_string.append("%s = %s" % (sk, semantic_values[condition_vec[i]]))
                    semantic_vector_string = ', '.join(semantic_vector_string)
                    if len(conditioned_raster):
                        print("\t\t\t(%s):\tSelected %u time bin out of %u, divided into %u trials - %u neurons"
                              % (semantic_vector_string, conditioned_raster.shape[0], len(array),
                                 len(np.unique(conditioned_trial)), conditioned_raster.shape[1]))
                    else:
                        print("\t\t\t(%s):\tNo data found" % semantic_vector_string)

            session_conditioned_data = [r.shape[0] for r in list(session_conditioned_rasters.values())]
            session_conditioned_trials = [len(np.unique(c)) for c in list(session_conditioned_trial_index.values())]

            self._max_conditioned_data = max([self._max_conditioned_data, np.max(session_conditioned_data)])
            self._min_conditioned_data = min([self._min_conditioned_data, np.min(session_conditioned_data)])

            # if the session has enough data for each condition, append it to the main data dictionary

            if np.min(session_conditioned_data) >= self._min_data_per_condition and \
                    np.min(session_conditioned_trials) >= self._min_trials_per_condition:
                for cv in self._condition_vectors:
                    self.conditioned_rasters[string_bool(cv)].append(session_conditioned_rasters[string_bool(cv)])
                    self.conditioned_trial_index[string_bool(cv)].append(
                        session_conditioned_trial_index[string_bool(cv)])
                if self._verbose:
                    print('\n')
                self.n_brains += 1
                self.n_neurons += list(session_conditioned_rasters.values())[0].shape[1]
                self.which_brain.append(np.ones(list(session_conditioned_rasters.values())[0].shape[1]) * self.n_brains)
            else:
                if self._verbose:
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
                semantic_keys.append(self._semantic_keys[np.where(col_sum == len(dic[0]))[0][0]])
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

    def all_dichotomies(self, balanced=True, semantic_names=False):
        if balanced:
            dichotomies = {}
            sem, keys = self._find_semantic_dichotomies()
            nsem = self._find_nonsemantic_dichotomies()
            if (self.n_conditions == 2) and semantic_names:
                dichotomies[keys[0]] = sem[0]
                dichotomies[keys[1]] = sem[1]
                dichotomies['XOR'] = nsem[0]
            else:
                for i in range(len(sem)):
                    dichotomies[keys[i]] = sem[i]
                for dic in nsem:
                    dichotomies[_powerchotomy_to_key(dic)] = dic
        else:
            powerchotomies = self._powerchotomies()
            dichotomies = {}
            for dk in powerchotomies:
                k = self._dic_key(powerchotomies[dk])
                if k and semantic_names:
                    dichotomies[k] = powerchotomies[dk]
                else:
                    dichotomies[dk] = powerchotomies[dk]
            if self.n_conditions == 2:
                dichotomies['XOR'] = dichotomies['00_11_v_01_10']
                del dichotomies['00_11_v_01_10']
        return dichotomies

    def _powerchotomies(self):
        conditions = list(self._semantic_vectors.keys())
        powerset = list(chain.from_iterable(combinations(conditions, r) for r in range(1, len(conditions))))
        dichotomies = {}
        for i in range(len(powerset)):
            for j in range(i + 1, len(powerset)):
                if len(np.unique(powerset[i] + powerset[j])) == len(conditions):
                    if len(powerset[i] + powerset[j]) == len(conditions):
                        dic = [list(powerset[i]), list(powerset[j])]
                        dichotomies[_powerchotomy_to_key(dic)] = dic
        return dichotomies

    def _dic_key(self, dic):
        if len(dic[0]) == 2 ** (self.n_conditions - 1) and len(dic[1]) == 2 ** (self.n_conditions - 1):
            for i in range(len(dic)):
                d = [string_bool(x) for x in dic[i]]
                col_sum = np.sum(d, 0)
                if len(dic[0]) in col_sum:
                    return self._semantic_keys[np.where(col_sum == len(dic[0]))[0][0]]
        return 0

    def _dichotomy_from_key(self, key):
        dics, keys = self._find_semantic_dichotomies()
        if key in keys:
            dic = dics[np.where(np.asarray(keys) == key)[0][0]]
        else:
            raise RuntimeError(
                "\n[dichotomy_from_key] The specified key does not correspond to a semantic dichotomy. Check the key value.")

        return dic

    def _generate_semantic_vectors(self):
        for condition_vec in self._condition_vectors:
            semantic_vector = '('
            for i, sk in enumerate(self._semantic_keys):
                semantic_values = list(self.conditions[sk])
                semantic_vector += semantic_values[condition_vec[i]] + ' '
            semantic_vector = semantic_vector[:-1] + ')'
            self._semantic_vectors[string_bool(condition_vec)] = semantic_vector

    def _compute_centroids(self):
        self.centroids = {w: np.hstack([np.nanmean(r, 0) for r in self.conditioned_rasters[w]])
                          for w in self.conditioned_rasters.keys()}

    def _zscore_activity(self):
        keys = [string_bool(w) for w in self._condition_vectors]
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
        if self._verbose:
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
            for n in range(self.n_brains):
                # select conditioned rasters
                for iteration in range(10):
                    all_conditions = list(self._semantic_vectors.keys())
                    all_data = np.vstack([self.conditioned_rasters[cond][n] for cond in all_conditions])
                    all_trials = np.hstack([self.conditioned_trial_index[cond][n] for cond in all_conditions])
                    all_n_trials = {cond: len(np.unique(self.conditioned_trial_index[cond][n])) for cond in
                                    all_conditions}

                    unique_trials = np.unique(all_trials)
                    np.random.shuffle(unique_trials)

                    i = 0
                    for cond in all_conditions:
                        cond_trials = unique_trials[i:i + all_n_trials[cond]]
                        new_cond_array = []
                        new_cond_trial = []
                        for trial in cond_trials:
                            new_cond_array.append(all_data[all_trials == trial])
                            new_cond_trial.append(all_trials[all_trials == trial])
                        self.conditioned_rasters[cond][n] = np.vstack(new_cond_array)
                        self.conditioned_trial_index[cond][n] = np.hstack(new_cond_trial)
                        i += all_n_trials[cond]

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
        if self._debug:
            print('\nCheck trial availability')
        for k in self.conditioned_trial_index:
            for i, ti in enumerate(self.conditioned_trial_index[k]):
                if self._debug:
                    print(k, 'raster %u:' % i, np.unique(ti).shape[0])
                    print(ti)
                if np.unique(ti).shape[0] < 2:
                    return False
        return True

    def _reset_weight_arrays(self):
        self.decoding_weights = {}
        self.decoding_weights_null = {}


# Wrapper for decoding

def decoding_analysis(data, conditions, decodanda_params, analysis_params, parallel=False, plot=False, ax=None):
    """
    Function that performs a balanced decoding analyses of the
    data set passed in the ``data`` argument, using variables and values
    specified in the ``conditions`` dictionary.

    This functions is a shortcut for building a ``Decodanda`` object
    with ``decodanda_params`` as arguments and calling the ``Decodanda.decode`` function
    with ``analysis_params`` as arguments.


    See Also
    --------
    Decodanda

    Decodanda.decode


    Notes
    -----
    This function is equivalent to

    >>> Decodanda(data, conditions, **decodanda_params).decode(**analysis_params)


    Parameters
    ----------
    data
        The data set used by the ``Decodanda`` object.
    conditions
        The conditions dictionary for the ``Decodanda`` object.
    decodanda_params
        A dictionary specifying the values for the ``Decodanda`` constructor parameters.
    analysis_params
        A dictionary specifying the values for the ``Decodanda.decode`` function parameters.
    parallel
        [Experimental] if ``True``, null model iterations are performed on separated threads.
    plot
        If ``True``, the decoding results are shown in a figure.
    ax
        If specified, and ``plot=True`` the results are shown in the specified axis.

    Returns
    -------
        performances, null


    """
    an_params = copy.deepcopy(analysis_params)

    if parallel:
        # Data
        null_iterations = an_params['nshuffles']
        an_params['nshuffles'] = 0
        performances, _ = Decodanda(data, conditions, **decodanda_params).decode(**an_params)

        # Null
        del an_params['nshuffles']
        pool = Pool()
        null_performances = pool.map(_NullmodelIterator(data, conditions, decodanda_params, an_params),
                                     range(null_iterations))
        null = {key: np.stack([p[key] for p in null_performances]) for key in null_performances[0].keys()}
    else:
        performances, null = Decodanda(data, conditions, **decodanda_params).decode(**an_params)

    if plot:
        plot_perfs_null_model(performances, null, ax=ax, ptype='zscore')
    return performances, null


# Utilities


def check_session_requirements(session, conditions, **decodanda_params):
    d = Decodanda(session, conditions, fault_tolerance=True, **decodanda_params)
    if d.n_brains:
        return True
    else:
        return False


def check_requirements_two_conditions(sessions, conditions_1, conditions_2, **decodanda_params):
    good_sessions = []
    for s in sessions:
        if check_session_requirements(s, conditions_1, **decodanda_params) and check_session_requirements(s,
                                                                                                          conditions_2,
                                                                                                          **decodanda_params):
            good_sessions.append(s)
    return good_sessions


def balance_decodandas(ds):
    for i in range(len(ds)):
        for j in range(i + 1, len(ds)):
            _balance_two_decodandas(ds[i], ds[j])


def _balance_two_decodandas(d1, d2, sampling_strategy='random'):
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

            if d1._verbose:
                print("Balancing data for d1: %u, d2: %u - now d1: %u, d2: %u" % (
                    t1, t2, list(d1.conditioned_rasters.values())[i][n].shape[0],
                    list(d2.conditioned_rasters.values())[i][n].shape[0]))

    for w in d1.conditioned_rasters.keys():
        d1.ordered_conditioned_rasters[w] = d1.conditioned_rasters[w].copy()
        d1.ordered_conditioned_trial_index[w] = d1.conditioned_trial_index[w].copy()
        d2.ordered_conditioned_rasters[w] = d2.conditioned_rasters[w].copy()
        d2.ordered_conditioned_trial_index[w] = d2.conditioned_trial_index[w].copy()

    print("\n")


def _generate_binary_condition(var_key, value1, value2, key1=None, key2=None, var_key_plot=None):
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


def _generate_binary_conditions(discrete_dict):
    conditions = {}
    for key in discrete_dict.keys():
        conditions[key] = {
            '%s' % discrete_dict[key][0]: lambda d, k=key: getattr(d, k) == discrete_dict[k][0],
            '%s' % discrete_dict[key][1]: lambda d, k=key: getattr(d, k) == discrete_dict[k][1],
        }
    return conditions


def _powerchotomy_to_key(dic):
    return '_'.join(dic[0]) + '_v_' + '_'.join(dic[1])


class _NullmodelIterator(object):  # necessary for parallelization of null model iterations
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
        if 'non_semantic' in self.analysis_params.keys():
            if self.analysis_params['non_semantic'] and len(self.conditions) == 2:
                semantic_dics.append([['01', '10'], ['00', '11']])
                semantic_keys.append('XOR')
            if self.analysis_params['non_semantic'] and len(self.conditions) > 2:
                dics = dec.all_dichotomies(balanced=True)
                semantic_dics = list(dics.values())
                semantic_keys = list(dics.keys())

        perfs = {}
        for key, dic in zip(semantic_keys, semantic_dics):
            if dec._verbose:
                print("\nTesting null decoding performance for semantic dichotomy: ", key)
            dec._shuffle_conditioned_arrays(dic)
            performance = dec.decode_dichotomy(dic, **self.analysis_params)
            perfs[key] = np.nanmean(performance)
            dec._order_conditioned_rasters()

        return perfs
