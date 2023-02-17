import numpy as np
import scipy.stats

from .imports import *


# Classes


class CrossValidator(object):
    # necessary for parallelization of cross validation repetitions
    # TODO: fix parallelization by fixing this
    def __init__(self, classifier, conditioned_rasters, conditioned_trial_index, dic, training_fraction, ndata, subset,
                 semantic_vectors):
        self.classifier = classifier
        self.conditioned_rasters = deepcopy(conditioned_rasters)
        self.conditioned_trial_index = deepcopy(conditioned_trial_index)
        self.dic = dic
        self.training_fraction = training_fraction
        self.ndata = ndata
        self.subset = subset
        self.semantic_vectors = semantic_vectors

    def __call__(self, i):
        self.i = i
        self.randomstate = RandomState(i)
        performance = self.one_cv_step(self.dic, self.training_fraction, self.ndata)
        return performance

    def train(self, training_raster_A, training_raster_B, label_A, label_B):

        training_labels_A = np.repeat(label_A, training_raster_A.shape[0]).astype(object)
        training_labels_B = np.repeat(label_B, training_raster_B.shape[0]).astype(object)

        training_raster = np.vstack([training_raster_A, training_raster_B])
        training_labels = np.hstack([training_labels_A, training_labels_B])

        self.classifier = sklearn.base.clone(self.classifier)

        training_raster = training_raster[:, self.subset]
        self.classifier.fit(training_raster, training_labels)

    def test(self, testing_raster_A, testing_raster_B, label_A, label_B):

        testing_labels_A = np.repeat(label_A, testing_raster_A.shape[0]).astype(object)
        testing_labels_B = np.repeat(label_B, testing_raster_B.shape[0]).astype(object)

        testing_raster = np.vstack([testing_raster_A, testing_raster_B])
        testing_labels = np.hstack([testing_labels_A, testing_labels_B])

        testing_raster = testing_raster[:, self.subset]
        performance = self.classifier.score(testing_raster, testing_labels)
        return performance

    def one_cv_step(self, dic, training_fraction, ndata):
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
            training, testing = sample_training_testing_from_rasters(self.conditioned_rasters[d],
                                                                     ndata,
                                                                     training_fraction,
                                                                     self.conditioned_trial_index[d],
                                                                     randomstate=self.randomstate)
            training_array_A.append(training)
            testing_array_A.append(testing)

        for d in set_B:
            training, testing = sample_training_testing_from_rasters(self.conditioned_rasters[d],
                                                                     ndata,
                                                                     training_fraction,
                                                                     self.conditioned_trial_index[d])
            training_array_B.append(training)
            testing_array_B.append(testing)

        training_array_A = np.vstack(training_array_A)
        training_array_B = np.vstack(training_array_B)
        testing_array_A = np.vstack(testing_array_A)
        testing_array_B = np.vstack(testing_array_B)

        self._train(training_array_A, training_array_B, label_A, label_B)

        performance = self._test(testing_array_A, testing_array_B, label_A, label_B)

        return performance


class DictSession:
    """
    Translator from dictionary to session object with getattr

    """

    def __init__(self, dictionary):
        for key in list(dictionary.keys()):
            self.__setattr__(key, np.asarray(dictionary[key]))

    def __getitem__(self, key):
        return self.__getattribute__(key)


class FakeSession:
    def __init__(self, n_neurons, ndata, persistence_letter=0.97, persistence_number=0.97, persistence_color=0.97,
                 noise_amplitude=0.5, coding_fraction=0.1, rotate=False, symplex=False):
        from sklearn.datasets import make_spd_matrix
        from scipy.stats import unitary_group
        self.raster = np.zeros((ndata, n_neurons))
        self.behaviour_letter = np.zeros(ndata, dtype=object)
        self.behaviour_number = np.zeros(ndata, dtype=object)
        self.behaviour_color = np.zeros(ndata, dtype=object)
        self.trial = np.arange(ndata)
        self.timebin = 1.0
        self.name = 'SessioneFinta1.0'
        n_let = int(n_neurons / 3)
        n_num = int(n_neurons / 3)
        n_col = n_neurons - n_let - n_num

        # define two multivariate gaussians as generative models for behaviour 'letter'
        means_A = np.random.rand(n_let)
        means_B = np.random.rand(n_let)
        cov_A = make_spd_matrix(n_let)
        cov_B = make_spd_matrix(n_let)

        # same thing for the other variable 'number'
        means_1 = np.random.rand(n_num)
        means_2 = np.random.rand(n_num)
        cov_1 = make_spd_matrix(n_num)
        cov_2 = make_spd_matrix(n_num)

        # same thing for the other variable 'colors'
        means_r = np.random.rand(n_col)
        means_g = np.random.rand(n_col)
        cov_r = make_spd_matrix(n_col)
        cov_g = make_spd_matrix(n_col)

        # fill arrays with generated data
        self.behaviour_letter[0] = 'A'
        self.behaviour_color[0] = 'red'
        self.behaviour_number[0] = 1

        for i in range(1, ndata):
            vector = np.zeros(n_neurons)

            if np.random.rand() < persistence_letter:
                if self.behaviour_letter[i - 1] == 'A':
                    vector[0:n_let] = np.random.multivariate_normal(means_A, cov_A, 1)
                    self.behaviour_letter[i] = 'A'
                if self.behaviour_letter[i - 1] == 'B':
                    vector[0:n_let] = np.random.multivariate_normal(means_B, cov_B, 1)
                    self.behaviour_letter[i] = 'B'
            else:
                if self.behaviour_letter[i - 1] == 'B':
                    vector[0:n_let] = np.random.multivariate_normal(means_A, cov_A, 1)
                    self.behaviour_letter[i] = 'A'
                if self.behaviour_letter[i - 1] == 'A':
                    vector[0:n_let] = np.random.multivariate_normal(means_B, cov_B, 1)
                    self.behaviour_letter[i] = 'B'

            if np.random.rand() < persistence_number:
                if self.behaviour_number[i - 1] == 1:
                    vector[n_let:n_let + n_num] = np.random.multivariate_normal(means_1, cov_1, 1)
                    self.behaviour_number[i] = 1
                if self.behaviour_number[i - 1] == 2:
                    vector[n_let:n_let + n_num] = np.random.multivariate_normal(means_2, cov_2, 1)
                    self.behaviour_number[i] = 2
            else:
                if self.behaviour_number[i - 1] == 2:
                    vector[n_let:n_let + n_num] = np.random.multivariate_normal(means_1, cov_1, 1)
                    self.behaviour_number[i] = 1
                if self.behaviour_number[i - 1] == 1:
                    vector[n_let:n_let + n_num] = np.random.multivariate_normal(means_2, cov_2, 1)
                    self.behaviour_number[i] = 2

            if np.random.rand() < persistence_color:
                if self.behaviour_color[i - 1] == 'red':
                    vector[n_let + n_num:] = np.random.multivariate_normal(means_r, cov_r, 1)
                    self.behaviour_color[i] = 'red'
                if self.behaviour_color[i - 1] == 'green':
                    vector[n_let + n_num:] = np.random.multivariate_normal(means_g, cov_g, 1)
                    self.behaviour_color[i] = 'green'
            else:
                if self.behaviour_color[i - 1] == 'green':
                    vector[n_let + n_num:] = np.random.multivariate_normal(means_r, cov_r, 1)
                    self.behaviour_color[i] = 'red'
                if self.behaviour_color[i - 1] == 'red':
                    vector[n_let + n_num:] = np.random.multivariate_normal(means_g, cov_g, 1)
                    self.behaviour_color[i] = 'green'

            self.raster[i] = vector

        # rotate the representations
        if rotate:
            M = unitary_group.rvs(n_neurons)
            for i in range(ndata):
                self.raster[i, :] = np.dot(self.raster[i, :], M)

        # add noise to the sampled array
        xi = np.random.rand(ndata, n_neurons) - 0.5
        self.raster = self.raster + noise_amplitude * xi * np.max(np.abs(self.raster))

        # add non-linearity by thresholding the raster
        threshold = np.percentile(np.abs(self.raster), 100 * (1 - coding_fraction))
        self.raster[np.abs(self.raster) < threshold] = 0
        self.raster[np.abs(self.raster) >= threshold] = 1

        # add dimensionality by resampling the blob centers
        if symplex:
            for color in ['red', 'green']:
                for letter in ['A', 'B']:
                    for number in [1, 2]:
                        mask = (self.behaviour_color == color) & (self.behaviour_letter == letter) & (
                                self.behaviour_number == number)
                        raster = self.raster[mask, :]
                        average = np.nanmean(raster, 0)
                        resampled_raster = raster - average + np.random.rand(n_neurons) - 0.5
                        # index = np.arange(n_neurons)
                        # np.random.shuffle(index)
                        # resampled_raster = resampled_raster[:, index]
                        self.raster[mask] = resampled_raster


class Logger:
    def __init__(self, filename=None):
        self.filename = filename
        self.writer = None

    def initialize(self, filename=None):
        if filename is not None:
            self.filename = filename
        self.writer = open(self.filename, 'w')
        self.writer.write('%s\n' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.writer.close()

    def log(self, string):
        self.writer = open(self.filename, 'a')
        self.writer.write(string + '\n')
        self.writer.close()

    def log_stats_nullmodel(self, key, data, null, p):
        logtext = f"{key: <25}{'Data: %.3f' % data : <16}{'Null: mean %.3f std %.3f n=%u' % (np.nanmean(null), np.nanstd(null), len(null)): <40}{p_to_text(p) : ^10}"
        self.log(logtext)

    def log_stats_ttest_1s(self, key, data, null, t, p):
        logtext = f"{key: <25}{'mean %.3f std %.3f n=%u' % (np.nanmean(data), np.nanstd(data), np.sum(np.isnan(data) == 0)): <30}{'T-test vs %.2f t=%.2f' % (null, t) : <25}{p_to_text(p):^10}"
        self.log(logtext)

    def log_stats(self, key, data, test, stat_name, stat_val, p):
        logtext = f"{key: <25}{'mean %.3f std %.3f n=%u' % (np.nanmean(data), np.nanstd(data), np.sum(np.isnan(data) == 0)): <30}{'%s %s=%.2f' % (test, stat_name, stat_val) : <35}{p_to_text(p):^10}"
        self.log(logtext)


# Sampling functions


def sample_from_rasters(rasters, ndata, mode='sample'):
    sampled_rasters = []
    if mode == 'sample':
        for r in rasters:
            sampling_index = np.random.randint(0, r.shape[0], ndata)
            sampled_rasters.append(r[sampling_index, :])
            raster = np.hstack(sampled_rasters)
    if mode == 'cut':
        m = np.min([r.shape[0] for r in rasters])
        raster = np.hstack([r[:m, :] for r in rasters])
    return raster


def sample_training_testing_from_rasters(rasters, ndata, training_fraction, trials, mode='sample',
                                         testing_trials=None, randomstate=None, debug=False):
    training_rasters = []
    testing_rasters = []

    if mode == 'sample':
        for i, r in enumerate(rasters):
            training_mask, testing_mask = training_test_block_masks(T=r.shape[0],
                                                                    training_fraction=training_fraction,
                                                                    trials=trials[i],
                                                                    randomstate=randomstate,
                                                                    debug=debug,
                                                                    testing_trials=testing_trials)
            training_r = r[training_mask, :]
            testing_r = r[testing_mask, :]
            # if debug:
            #     f, axs = plt.subplots(2, 1, figsize=(7, 6))
            #     visualize_raster(r, ax=axs[0])
            #     axs[1].plot(training_mask, label='training')
            #     axs[1].plot(testing_mask, label='testing')
            #     axs[1].plot(trials[i] / np.max(trials[i]), color='k')
            #     plt.legend()
            #     axs[1].set_xlabel('time')
            #     axs[1].set_title('raster %u' % i)

            if debug:
                print("\t Raster %u: Using %u data from %u trials for training, %u data from %u trials for testing" % (
                    i, np.sum(training_mask), len(np.unique(trials[i][training_mask])),
                    np.sum(testing_mask), len(np.unique(trials[i][testing_mask]))
                ))

            if randomstate is None:
                sampling_index_training = np.random.randint(0, np.sum(training_mask), ndata)
                sampling_index_testing = np.random.randint(0, np.sum(testing_mask), ndata)
            else:
                sampling_index_training = randomstate.randint(0, np.sum(training_mask), ndata)
                sampling_index_testing = randomstate.randint(0, np.sum(testing_mask), ndata)
            training_rasters.append(training_r[sampling_index_training, :])
            testing_rasters.append(testing_r[sampling_index_testing, :])

    training_raster = np.hstack(training_rasters)
    testing_raster = np.hstack(testing_rasters)

    return training_raster, testing_raster


def training_test_block_masks(T, training_fraction, trials, randomstate=None, debug=False, testing_trials=None):
    if training_fraction == 1:
        return np.ones(T) > 0, np.zeros(T) > 0
    if training_fraction == 0:
        return np.zeros(T) > 0, np.ones(T) > 0

    unique_trial_numbers = np.unique(trials)
    nutn = len(unique_trial_numbers)
    if testing_trials is None:
        if (nutn * (1. - training_fraction)) <= 1:  # in the case of very small number of trials, use all but one for training
            if randomstate is None:
                testing_trials = unique_trial_numbers[np.random.choice(nutn, size=1, replace=False)]
            else:
                testing_trials = unique_trial_numbers[randomstate.choice(nutn, size=1, replace=False)]
        else:
            if randomstate is None:
                testing_trials = unique_trial_numbers[
                    np.random.choice(nutn, size=ceil(nutn * (1. - training_fraction)), replace=False)]
            else:
                testing_trials = unique_trial_numbers[
                    randomstate.choice(nutn, size=ceil(nutn * (1. - training_fraction)), replace=False)]
        if debug:
            print("All trials:", unique_trial_numbers)
            print("Testing trials:", testing_trials)
    testing_mask = np.in1d(trials, testing_trials)
    training_mask = testing_mask == 0
    return training_mask, testing_mask


def chunk_shuffle_index(T, chunk_size):
    index = np.arange(T)
    N = int(T / chunk_size)
    temp_index = index[:int(N * chunk_size)]
    remaining_index = index[int(N * chunk_size):]

    # shuffling in chunks
    temp_index = temp_index.reshape(-1, chunk_size)
    np.random.shuffle(temp_index)
    temp_index = temp_index.flatten()

    # inserting the last missing piece in a random position
    random_position = int(np.random.rand() * N * chunk_size)

    shuffled_index = np.concatenate((temp_index[:random_position], remaining_index, temp_index[random_position:]))
    return shuffled_index


def contiguous_chunking(mask, max_chunk_size=None):
    if np.sum(mask):
        chunk_index = np.nan * np.zeros(len(mask))
        trial_begins = np.where(np.diff(mask.astype(int)) == 1)[0] + 1
        trial_ends = np.where(np.diff(mask.astype(int)) == -1)[0]

        # if there is only a begin and no end, then len(mask) is the end
        if len(trial_begins) and not len(trial_ends):
            trial_ends = np.hstack([trial_ends, np.asarray([len(mask)])])
        # if there is only and end and not a beginning, then 0 is the start
        if len(trial_ends) and not len(trial_begins):
            trial_begins = np.hstack([np.asarray([0]), trial_begins])
        # if the first end comes before the first begin, then we are missing a 0 at the beginning
        if trial_ends[0] < trial_begins[0]:
            trial_begins = np.hstack([np.asarray([0]), trial_begins])
        # if the last begin comes after the last end, then we are missing a len(mask) at the end
        if trial_ends[-1] < trial_begins[-1]:
            trial_ends = np.hstack([trial_ends, np.asarray([len(mask)])])

        # now chunking
        if max_chunk_size is not None:
            break_to_insert = []
            where_to_insert = []
            for i, pair in enumerate(zip(trial_begins, trial_ends)):
                n_breaks = floor((pair[1] - pair[0]) / (max_chunk_size + 1))
                if n_breaks:
                    for k in range(n_breaks):
                        where_to_insert.append(i)
                        break_to_insert.append(pair[0] + (k + 1) * max_chunk_size - 1)
            if len(where_to_insert):
                trial_ends = np.insert(trial_ends, np.asarray(where_to_insert), break_to_insert)
                trial_begins = np.insert(trial_begins, np.asarray(where_to_insert) + 1, np.asarray(break_to_insert) + 1)

        for i, pair in enumerate(zip(trial_begins, trial_ends)):
            chunk_index[pair[0]:pair[1] + 1] = i
    else:
        chunk_index = np.repeat(np.nan, len(mask))
    return chunk_index


def non_contiguous_mask(trials, chunks):
    mask = np.zeros(len(trials)).astype(bool)
    for ti in np.unique(trials):
        trial_chunks = np.unique(chunks[trials == ti])
        for i, ci in enumerate(trial_chunks):
            if i % 2 == 0:
                mask[chunks == ci] = True
    return mask


# Decodanda init utilities

def delete_silent_bins(array):
    mask = np.sum(array, 1) > 0
    return array[mask, :]


def string_bool(x):
    if (type(x) == str) or (type(x) == np.str_):
        values = []
        for s in x:
            values.append(bool(int(s)))
    elif (type(x) == list) or (type(x)==type(np.zeros(10))):
        values = ''
        for s in x:
            values += str(s)
    else:
        print(x, type(x))
        print('WE HAVE A PROBLEM')
        raise ValueError('x (type=%s) should either be a string, a list, or a numpy array' % type(x))
    return values


def generate_binary_words(n):
    words = list(itertools.product([0, 1], repeat=n))
    words = [list(w) for w in words]
    return np.asarray(words)


def generate_dichotomies(n):
    words = generate_binary_words(n)
    dy_words = generate_binary_words(2 ** n)
    dichotomies = []
    sets = []
    for w in dy_words:
        string_w = string_bool(w)
        string_not_w = string_bool((w == 0).astype(int))
        if (np.sum(w) == np.sum(w == 0)) and (string_not_w not in dichotomies):
            dichotomies.append(string_w)
            set_A = [string_bool(x) for x in words[w > 0]]
            set_B = [string_bool(x) for x in words[w == 0]]
            sets.append([set_A, set_B])
    return dichotomies, sets


def semantic_score(dic):
    d = [string_bool(x) for x in dic[0]]
    fingerprint = np.abs(np.sum(d, 0) - len(d) / 2)
    return np.max(fingerprint) * np.sum(fingerprint)


# Analysis

def hamming_distance(x, y):
    d = [x[i] != y[i] for i in range(len(x))]
    return np.sum(d)


def hamming(x, y):
    if type(x) == str:
        return np.sum((np.asarray(string_bool(x)) == np.asarray(string_bool(y))) == 0)
    else:
        return np.sum((np.asarray(x) == np.asarray(y)) == 0)


def cosine(x, y):
    return np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y))


def distance_from_plane(p1, p2, p3, point):
    def distance(x):
        v = p1 + x[0] * (p2 - p1) + x[1] * (p3 - p1)
        d = point - v
        return np.sqrt(np.dot(d, d))

    from scipy import optimize
    res = optimize.minimize(distance, np.asarray([1, 1]), options={'disp': False})

    d = distance(res.x)

    interdistances = [
        np.sqrt(np.dot(p1 - p2, p1 - p2)),
        np.sqrt(np.dot(p2 - p3, p2 - p3)),
        np.sqrt(np.dot(p1 - p3, p1 - p3)),
    ]
    interd = np.nanmean(interdistances)
    return d / interd


def interplanar_distance(centroids):
    interdistances = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            v = centroids[i] - centroids[j]
            interdistances.append(np.sqrt(np.dot(v, v)))

    md = np.sqrt(6) / 3.

    planar_distances = [distance_from_plane(centroids[0], centroids[1], centroids[2], centroids[3]) / md,
                        distance_from_plane(centroids[1], centroids[2], centroids[3], centroids[0]) / md,
                        distance_from_plane(centroids[2], centroids[3], centroids[0], centroids[1]) / md,
                        distance_from_plane(centroids[3], centroids[0], centroids[1], centroids[2]) / md]

    return planar_distances


# Statistics


def z_pval(x, null):
    from scipy.stats import norm
    std = np.nanstd(null)
    mean = np.nanmean(null)
    z_score = (x - mean) / std
    p_value = norm.sf(abs(z_score))
    return z_score, p_value


def p_to_ast(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    if p >= 0.05:
        return 'ns'


def p_to_text(p):
    if p < 0.0001:
        return '***\nP=%.1e' % p
    if p < 0.001:
        return '***\nP=%.4f' % p
    if p < 0.01:
        return '**\nP=%.3f' % p
    if p < 0.05:
        return '*\nP=%.3f' % p
    if p >= 0.05:
        return 'ns\nP=%.2f' % p


def count_pval(x, null):
    p = np.nanmean(np.asarray(null) > x)
    return p


# Visualization #


def metric_dissimilarity(X):
    return cdist(X, X)


def corr_dissimilarity(X):
    ndim = X.shape[0]
    D = np.zeros((ndim, ndim))
    for i in range(ndim):
        for j in range(i + 1, ndim):
            D[i, j] = 1 - scipy.stats.pearsonr(X[i], X[j])[0]
            D[j, i] = D[i, j]
    print(D.shape)
    return D


def cosyne_dissimilarity(X):
    ndim = X.shape[0]
    D = np.zeros((ndim, ndim))
    for i in range(ndim):
        for j in range(i + 1, ndim):
            D[i, j] = 1. / cossim(X[i], X[j])
            D[j, i] = D[i, j]
    print(D.shape)
    return D


def cossim(x, y):
    return np.dot(x, y) / (np.dot(x, x) * np.dot(y, y))


def mahalanobis_dissimilarity(d):
    semantic_keys = d._semantic_vectors.keys()

    # compute block covariance matrix
    Cs = []
    for k in semantic_keys:
        Cs.append(block_diagonal([np.cov(r.transpose()) for r in d.conditioned_rasters[k]]))

    rs = []
    for k in semantic_keys:
        rs.append(d.centroids[k])

    ndim = len(semantic_keys)
    D = np.zeros((ndim, ndim))
    # compute mahalanobis
    for i in range(ndim):
        for j in range(i + 1, ndim):
            D[i, j] = 0.5 * (mahalanobis(rs[i], rs[j], Cs[i]) + mahalanobis(rs[i], rs[j], Cs[j]))
            D[j, i] = D[i, j]

    return D


def block_diagonal(arrs):
    shapes = np.array([a.shape[0] for a in arrs])
    out = np.zeros((np.sum(shapes), np.sum(shapes)))

    r, c = 0, 0
    for i, rr in enumerate(shapes):
        out[r:r + rr, r:r + rr] = arrs[i]
        r += rr
    return out


def equalize_ax(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    From karlo on stack exchange
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def log_dichotomy(dec, dic, ndata, s='Decoding'):
    set_A = dic[0]
    label_A = ''
    for d in set_A:
        label_A += (dec._semantic_vectors[d] + ' ')
    label_A = label_A[:-1]

    set_B = dic[1]
    label_B = ''
    for d in set_B:
        label_B += (dec._semantic_vectors[d] + ' ')
    label_B = label_B[:-1]
    if dec.n_conditions <= 2:
        print(
            '\n[decode_dichotomy]\t%s - %u time bins - %u neurons - %u brains'
            '\n\t\t%s\n\t\t\tvs.\n\t\t%s'
            % (s, ndata, len(dec.subset), dec.n_brains, label_A, label_B))

    if dec.n_conditions > 2:
        print(
            '\n[decode_dichotomy]\t%s - %u time bins - %u neurons - %u brains'
            '\n\t\t%s\n\t\t\t\t\tvs.\t\t\t\t\t\n\t\t%s'
            % (s, ndata, len(dec.subset), dec.n_brains, label_A, label_B))


def destroy_time_correlations(array):
    # array is TxN features
    new_array = np.transpose(array)
    [np.random.shuffle(x) for x in new_array]
    array = np.transpose(new_array)


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

    allmax = np.nanmax(raster)
    for i in range(raster.shape[1]):
        ax.plot(raster[:, order[i]] / allmax * 2 + i)
    return ax


def visualize_data_vs_null(data, null, value, ax=None):
    # computing the P value of the z-score
    from scipy.stats import norm
    null_mean = np.nanmean(null)
    z = (data - null_mean) / np.nanstd(null)
    p = norm.sf(abs(z))

    def p_to_ast(p):
        if p < 0.001:
            return '***'
        if p < 0.01:
            return '**'
        if p < 0.05:
            return '*'
        if p >= 0.05:
            return 'ns'

    # visualizing
    if ax is None:
        f, ax = plt.subplots(figsize=(6, 3))
    kde = scipy.stats.gaussian_kde(null)
    null_x = np.linspace(np.nanmean(null)-5*np.nanstd(null), np.nanmean(null)+5*np.nanstd(null), 100)
    null_y = kde(null_x)
    ax.plot(null_x, null_y, color='k', alpha=0.5)
    ax.fill_between(null_x, null_y, color='k', alpha=0.3)
    ax.text(null_x[np.argmax(null_y)], np.max(null_y) * 1.05, 'null model', ha='center')
    sns.despine(ax=ax)
    ax.plot([data, data], [0, np.max(null_y)], color='red', linewidth=3)
    ax.text(data, np.max(null_y) * 1.05, 'data', ha='center', color='red')
    ax.set_xlabel(value)
    if data < np.nanmean(null):
        ax.text(0.85, 0.95, '%s\nz=%.1f\nP=%.1e' % (p_to_ast(p), z, p), ha='center', transform=ax.transAxes)
    else:
        ax.text(0.15, 0.95, '%s\nz=%.1f\nP=%.1e' % (p_to_ast(p), z, p), ha='center', transform=ax.transAxes)

    ax.plot(null, np.zeros(len(null)), linestyle='', marker='|', color='k')
    _ = ax.plot([null_mean, null_mean], [0, kde(null_mean)], color='k', linestyle='--')
    return z, p


# Histogram comparison

def histogram_comparison(Adata, Bdata, labelA, labelB, quantity, bins=None, ax=None):
    A = Adata[np.isnan(Adata) == 0]
    B = Bdata[np.isnan(Bdata) == 0]

    allmax = np.max([np.nanmax(A), np.nanmax(B)])
    allmin = np.min([np.nanmin(A), np.nanmin(B)])
    if type(bins) == int:
        bins = np.linspace(allmin, allmax, bins)
    if ax is None:
        f, ax = plt.subplots(figsize=(6, 4))

    ax.hist(A, bins=bins, color=pltcolors[0], alpha=0.3,
            label=labelA, density=True, linewidth=3)
    ax.hist(B, bins=bins, color=pltcolors[1], alpha=0.3,
            label=labelB, density=True, linewidth=3)

    sns.despine(ax=ax)
    ax.set_xlabel(quantity)
    ax.set_ylabel('Probability')

    if bins is None:
        bins = np.linspace(allmin, allmax, bins)
    kdeA = scipy.stats.gaussian_kde(A)
    yA = kdeA(bins)
    ax.plot(bins, yA, color=pltcolors[0], linewidth=2, alpha=0.8)

    kdeB = scipy.stats.gaussian_kde(B)
    yB = kdeB(bins)
    ax.plot(bins, yB, color=pltcolors[1], linewidth=2, alpha=0.8)

    ax.legend()

    ax.plot([np.nanmean(A), np.nanmean(A)], [0, kdeA(np.nanmean(A))], color=pltcolors[0], linewidth=2, linestyle='--')
    ax.text(np.nanmean(A), kdeA(np.nanmean(A)) * 1.1, 'mean=%.2f' % np.nanmean(A), color=pltcolors[0])

    ax.plot([np.nanmean(B), np.nanmean(B)], [0, kdeB(np.nanmean(B))], color=pltcolors[1], linewidth=2, linestyle='--')
    ax.text(np.nanmean(B), kdeB(np.nanmean(B)) * 1.1, 'mean=%.2f' % np.nanmean(B), color=pltcolors[1])


# Box comparison 2 suite

def draw_pair_plot(data1, data2, x1, x2, ax, swarm=False):
    for i in range(len(data1)):
        ax.plot([x1, x2], [data1[i], data2[i]], color=pltcolors[0], alpha=0.8, linewidth=1)
    if swarm:
        plt.errorbar([x1, x2], [np.nanmean(data1), np.nanmean(data2)],
                     [np.nanstd(data1) / np.sqrt(np.sum(np.isnan(data1)) == 0),
                      np.nanstd(data2) / np.sqrt(np.sum(np.isnan(data2)) == 0)], linestyle='', marker='o',
                     color=pltcolors[0])


def print_stats(data, name):
    d = np.asarray(data)
    m = np.nanmean(d)
    std = np.nanstd(d)
    stder = scipy.stats.sem(d[np.isnan(d) == 0])
    print("O> %s :\tmean %.3f +- %.3f SEM, std: %.3f" % (name, m, stder, std))


def annotate_ttest_p(dataA, dataB, x1, x2, ax, pairplot=False, force=False, p=-1, h='max'):
    if pairplot:
        data1 = np.asarray(dataA)[(np.isnan(dataA) == 0) & (np.isnan(dataB) == 0)]
        data2 = np.asarray(dataB)[(np.isnan(dataA) == 0) & (np.isnan(dataB) == 0)]
        draw_pair_plot(data1, data2, x1, x2, ax)
        if p == -1:
            p = scipy.stats.mstats_basic.ttest_rel(data1, data2)[1]
    else:
        data1 = np.asarray(dataA)[np.isnan(dataA) == 0]
        data2 = np.asarray(dataB)[np.isnan(dataB) == 0]
        if p == -1:
            p = scipy.stats.mstats_basic.ttest_ind(data1, data2)[1]

    if (p < 0.05) or (force):
        if h == 'max':
            ys = [np.min([np.min(data1), np.min(data2)]), np.max([np.max(data1), np.max(data2)])]
        if h == 'std':
            ys = [np.min([np.nanmean(data1) - np.nanstd(data1), np.nanmean(data2) - np.nanstd(data2)]),
                  np.max([np.nanmean(data1) + np.nanstd(data1), np.nanmean(data2) + np.nanstd(data2)])]
        y = ys[1] + (ys[1] - ys[0]) * 0.25
        dy = (ys[1] - ys[0]) * 0.05
        dx = np.abs(x1 - x2) * 0.02
        ax.plot([x1 + dx, x1 + dx, x2 - dx, x2 - dx], [y, y + dy, y + dy, y], 'k')
        ax.text((x1 + x2) / 2., y + 2 * dy, p_to_text(p), ha='center', va='bottom', color='k', fontsize=7)
        # ax.set_ylim([ys[0], ys[1]+(ys[1]-ys[0])*0.15])

    return p


def box_comparison_two(A, B, labelA, labelB, quantity, force=False, swarm=False, violin=False, box=False, paired=False,
                       bar=False, p=None, ax=None):
    print_stats(A, labelA)
    print_stats(B, labelB)
    if ax is None:
        f, ax = plt.subplots(figsize=(2.5, 3.5))
    dataA = pd.DataFrame(A, columns=[labelA])
    dataB = pd.DataFrame(B, columns=[labelB])
    data = pd.concat([dataA, dataB], sort='False')
    data = data[[labelA, labelB]]
    if box:
        sns.boxplot(data=data, ax=ax, width=0.4)
    elif bar:
        sns.barplot(data=data, ax=ax, capsize=.2, alpha=0.7)
    else:
        sns.pointplot(data=data, ax=ax, capsize=.2, join=False, alpha=0.5, marker='none', color='k')
    if swarm:
        if paired == 0:
            sns.swarmplot(data=data, ax=ax, color='0.5', alpha=0.6)
    if violin:
        sns.violinplot(data=data, ax=ax, color='0.5', alpha=0.3)
    if paired:
        draw_pair_plot(A, B, 0, 1, ax, swarm=swarm)

    ax.set_ylabel(quantity)
    if paired:
        nanmask = (np.isnan(A) == 0) & (np.isnan(B) == 0)
        pw = scipy.stats.wilcoxon(A[nanmask], B[nanmask])[1]
        pt = scipy.stats.ttest_rel(A[nanmask], B[nanmask])[1]

        print("Paired t-test (%s)-(%s):\t%s" % (labelA, labelB, p_to_text(pt)))
        print("Paired wilcoxon (%s)-(%s):\t%s" % (labelA, labelB, p_to_text(pw)))

        print("Number of data %s: %u, %s: %u\n" % (labelA, np.sum(nanmask), labelB, np.sum(nanmask)))
    else:
        if p is None:
            p1 = annotate_ttest_p(A, B, 0, 1, ax, force=force)
            print("Un-paired t-test (%s)-(%s):\t%s" % (labelA, labelB, p_to_text(p1)))
            print("Number of data %s: %u, %s: %u\n" % (
                labelA, np.sum(np.isnan(A) == 0), labelB, np.sum(np.isnan(B) == 0)))
        else:
            annotate_ttest_p(A, B, 0, 1, ax, force=force, p=p)
    sns.despine()

    return ax


# Synthetic data


def generate_synthetic_data(n_neurons=50, n_trials=50, timebins_per_trial=5, keyA='stimulus', keyB='action',
                           rateA=0.1, rateB=0.1, corrAB=0, scale=1, meanfr=0.1, mixing_factor=0., mixed_term=0.):

    session = {}

    # sample correlated labels
    labelsA = np.random.rand(n_trials) > 0.5
    labelsB = np.zeros(n_trials)
    trial = np.repeat(np.arange(n_trials), timebins_per_trial)

    for i in range(n_trials):
        if np.random.rand() >= corrAB:
            labelsB[i] = np.random.rand() > 0.5
        else:
            labelsB[i] = labelsA[i]

    # sample variables array
    behavior_A = np.repeat(labelsA, timebins_per_trial) * 2 - 1
    behavior_B = np.repeat(labelsB, timebins_per_trial) * 2 - 1

    # sample neural parameters
    rates = np.random.lognormal(meanfr, scale, n_neurons)
    rA = np.random.randn(n_neurons) * rateA
    rB = np.random.randn(n_neurons) * rateB
    rC = np.random.randn(n_neurons) * (rateB+rateA)/2.

    # sample activity
    raster = []
    for n in range(n_neurons):
        x = np.zeros(n_trials*timebins_per_trial)
        for A in [-1, 1]:
            for B in [-1, 1]:
                mask = (behavior_B == B) & (behavior_A == A)
                if n > n_neurons/2:
                    f = (1 - mixing_factor) * rA[n] * A + mixing_factor * rB[n] * B + rA[n] * (A*B) * mixed_term
                else:
                    f = (1 - mixing_factor) * rB[n] * B + mixing_factor * rA[n] * A + rB[n] * (A*B) * mixed_term
                f -= 2 * np.min(np.hstack([rA, rB]))
                if f < 0:
                    f = 0
                lam = rates[n] * f
                activity = np.random.poisson(lam=lam, size=np.sum(mask))
                x[mask] = activity
        raster.append(x)

    raster = np.asarray(raster).T

    session['raster'] = raster
    session[keyA] = behavior_A
    session[keyB] = behavior_B
    session['trial'] = trial

    return session


def visualize_synthetic_data(session):

    keys = [k for k in session.keys() if k != 'raster' and k != 'trial']
    frA0 = np.nanmean(session['raster'][session[keys[0]] == -1], 0)
    frA1 = np.nanmean(session['raster'][session[keys[0]] == 1], 0)
    frB0 = np.nanmean(session['raster'][session[keys[1]] == -1], 0)
    frB1 = np.nanmean(session['raster'][session[keys[1]] == 1], 0)
    selA = np.abs(frA1 - frA0)/(frA1 + frA0)
    selB = np.abs(frB1 - frB0)/(frB1 + frB0)
    order = np.argsort(selA - selB)

    f, axs = plt.subplots(4, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [6, 0.75, 1, 1]})
    activity = session['raster'][:, order]
    n_trials, n_neurons = session['raster'].shape
    axs[-1].set_xlabel('Time')
    axs[0].set_ylabel('Neuron #')
    sns.despine(f)

    for i in range(int(n_neurons)):
        xval = np.where(activity[:, i] > 0)[0]
        axs[0].scatter(xval, np.ones(len(xval))*i, color='k', marker='|', alpha=0.5)
        # nanact = np.copy(activity[:, i])
        # nanact[nanact == 0] = np.nan
        # axs[0].plot(i + (nanact>0), marker='|', linestyle='', color='k', alpha=0.5, markersize=3)

    axs[1].plot(session['trial'], color='k')
    axs[2].plot(session[keys[0]])
    axs[3].plot(session[keys[1]], color=pltcolors[1])

    axs[0].set_title('raster')
    axs[1].set_title('trial')
    axs[2].set_title(keys[0])
    axs[3].set_title(keys[1])


def generate_synthetic_data_intime(n_neurons=50, min_time=-10, max_time=10, signal=0.2, ntrials=10):
    rA = np.random.rand(n_neurons) * signal
    rB = np.random.rand(n_neurons) * signal
    rnone = np.random.rand(n_neurons) * signal
    data = {
        'raster': [],
        'trial': [],
        'time_from_onset': [],
        'stimulus': []
    }

    # sampling A trials
    for trial in range(ntrials):
        trial_raster = []
        trial_time = np.arange(min_time, max_time)
        for t in trial_time:
            vector = []
            if t > 0:
                for n in range(n_neurons):
                    vector.append(np.random.poisson(lam=rA[n], size=1))
            if t <= 0:
                for n in range(n_neurons):
                    vector.append(np.random.poisson(lam=rnone[n], size=1))
            trial_raster.append(vector)

        data['raster'].append(trial_raster)
        data['time_from_onset'].append(trial_time)
        data['trial'].append(np.ones(len(trial_time))*trial)
        data['stimulus'].append(np.repeat('A', len(trial_time)))

    # sampling B trials
    for trial in range(ntrials):
        trial_raster = []
        trial_time = np.arange(min_time, max_time)
        for t in trial_time:
            vector = []
            if t > 0:
                for n in range(n_neurons):
                    vector.append(np.random.poisson(lam=rB[n], size=1))
            if t <= 0:
                for n in range(n_neurons):
                    vector.append(np.random.poisson(lam=rnone[n], size=1))
            trial_raster.append(vector)

        data['raster'].append(trial_raster)
        data['time_from_onset'].append(trial_time)
        data['trial'].append(np.ones(len(trial_time)) * trial)
        data['stimulus'].append(np.repeat('B', len(trial_time)))

    data['raster'] = np.vstack(data['raster'])
    data['time_from_onset'] = np.hstack(data['time_from_onset'])
    data['trial'] = np.hstack(data['trial'])
    data['stimulus'] = np.hstack(data['stimulus'])

    return data

