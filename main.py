from sklearn.svm import SVC
from decodanda import *
from decodanda.utilities import FakeSession


from decodanda import Decodanda, FakeSession
from decodanda.visualize import visualize_raster
#
# We create a synthetic data set where neurons respond to two variables, labeled as
#  - behaviour_letter (taking values A, B)
#  - behaviour_number (taking values 1, 2)
# neural activity is stored under the 'raster' keyword

my_session = FakeSession(n_neurons=150,
                         ndata=500,
                         noise_amplitude=0.05,
                         coding_fraction=0.3,
                         rotate=True,
                         symplex=False)

dec = Decodanda(sessions=my_session,
                neural_attr='raster',
                conditions={
                    'behaviour_letter': ['A', 'B'],
                    'behaviour_number': [1, 2]
                })

s1 = FakeSession(n_neurons=150,
                 ndata=3000,
                 noise_amplitude=0.05,
                 coding_fraction=0.3,
                 rotate=True,
                 symplex=False)

conditions = {
        'number': {
            '1': lambda s: s.behaviour_number < 1.5,
            '2': lambda s: s.behaviour_number > 1.5
        },

        'letter': {
            'A': lambda s: s.behaviour_letter == 'A',
            'B': lambda s: s.behaviour_letter == 'B'
        }
    }

svc_2 = SVC(C=1.0, kernel='poly', degree=3, gamma=2)
mydec = Decodanda(sessions=s1, conditions=conditions, verbose=True, neural_attr='raster', classifier=svc_2)
mydec.decode(training_fraction=0.7, plot=True)
