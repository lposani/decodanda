import numpy as np
import matplotlib.pyplot as plt
from decodanda import Decodanda, FakeSession

# We create a synthetic data set where neurons respond to two variables, labeled as
#  - behaviour_letter (taking values A, B)
#  - behaviour_number (taking values 1, 2)
# neural activity is stored under the 'raster' keyword
#
# Through the geometry_analysis() function, here we test
# - the decoding() function
# - the CCGP() function
# - all the dichotomy logic

np.random.seed(0)

# Disentangled representations
s1 = FakeSession(n_neurons=120,
                 ndata=500,
                 noise_amplitude=0.02,
                 coding_fraction=0.3,
                 rotate=False,
                 symplex=False)

conditions = {
    'Stimulus': {
        'A': lambda s: s.behaviour_letter == 'A',
        'B': lambda s: s.behaviour_letter == 'B'
    },
}
mydec = Decodanda(data=s1,
                  conditions=conditions,
                  verbose=True)
mydec.decode(training_fraction=0.75, plot=True)
mydec.CCGP(plot=True)
