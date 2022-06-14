from sklearn.svm import SVC
from decodanda import *

# We create a synthetic data set where neurons respond to two variables, labeled as
#  - behaviour_letter (taking values A, B)
#  - behaviour_number (taking values 1, 2)
# neural activity is stored under the 'raster' keyword
# here we test
# - creating pseudo populations
# - the .decode() and .CCGP() functions with three variables
# - the ability of using polynomial classifiers instead of the default LinearSVC

s1 = FakeSession(n_neurons=110,
                 ndata=500,
                 noise_amplitude=0.05,
                 coding_fraction=0.3,
                 rotate=True,
                 symplex=False)

s2 = FakeSession(n_neurons=90,
                 ndata=600,
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
    },

    'color': {
        'r': lambda s: s.behaviour_color == 'red',
        'g': lambda s: s.behaviour_color == 'green'
    }
}

svc_2 = SVC(C=1.0, kernel='poly', degree=3, gamma=2)
mydec = Decodanda(data=[s1, s2],
                  conditions=conditions,
                  verbose=True,
                  classifier=svc_2)

mydec.decode(training_fraction=0.7, cross_validations=8, nshuffles=5, plot=True)
mydec.CCGP(plot=True, nshuffles=5)
plt.show()
