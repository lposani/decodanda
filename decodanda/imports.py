# standard stuff
import itertools
import types
from copy import deepcopy
from functools import partial
from math import ceil, floor
# parallelization stuff
from multiprocessing import Pool

# data reading stuff
import h5py
# visualization stuff
import matplotlib as mpl
# matrix stuff
import numpy as np
import pandas as pd
import scipy.stats
# learning stuff
import sklearn
from numpy.random import RandomState
from scipy.spatial.distance import cdist
from scipy.stats import linregress, pearsonr, ttest_1samp
from sklearn.svm import LinearSVC

mpl.rcParams.update({'figure.autolayout': True})
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

pltcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
pltcolors = pltcolors * 100
import datetime
from itertools import chain, combinations
from typing import Optional, Union

import matplotlib.transforms as transforms
import seaborn as sns
from matplotlib import animation, cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import mahalanobis
from scipy.stats import special_ortho_group
from sklearn.manifold import MDS
from tqdm import tqdm
