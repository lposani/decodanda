from decodanda import *

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

s1 = FakeSession(n_neurons=120,
                 ndata=2000,
                 noise_amplitude=0.02,
                 coding_fraction=0.3,
                 rotate=False,
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

mydec = Decodanda(data=s1,
                  conditions=conditions,
                  verbose=False)

mydec.geometrical_analysis(training_fraction=0.8, nshuffles=2, visualize=True)
mydec.geometrical_analysis(training_fraction=0.8, nshuffles=2, visualize=False)

plt.show()
plt.savefig('./Semantic_score.pdf')
