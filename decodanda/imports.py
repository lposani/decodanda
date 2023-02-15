# standard stuff
import types
from math import ceil, floor
import scipy.stats

# matrix stuff
import numpy as np
import itertools
import pandas as pd
from scipy.spatial.distance import cdist

# learning stuff
import sklearn
from scipy.stats import pearsonr
from scipy.stats import ttest_1samp
from scipy.stats import linregress
from sklearn.svm import LinearSVC

# parallelization stuff
from multiprocessing import Pool
from functools import partial
from copy import deepcopy
from numpy.random import RandomState

# visualization stuff
import matplotlib as mpl
mpl.rcParams.update({'figure.autolayout': True})
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
pltcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
pltcolors = pltcolors * 100
import seaborn as sns
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from scipy.spatial.distance import mahalanobis
from tqdm import tqdm
import datetime
from matplotlib import cm
from scipy.stats import special_ortho_group
import matplotlib.transforms as transforms
from typing import Union
from typing import Optional
from itertools import chain, combinations
