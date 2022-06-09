from decodanda import *
np.random.seed(1123)

# Dummy random data with one behavioural variable

session = {
    'Neural Data': np.random.rand(1000, 33),
    'Position': np.random.rand(1000) * 10
}

conditions = {                                      # Names of the variable and of the conditions can be anything,
    'Corner': {                                    # they are used for visualization purposes only
        'Top': lambda d: d['Position'] > 9,
        'Bottom': lambda d: d['Position'] < 1
    },
}

# Create a decodanda object by applying these conditions to the data

mydec = Decodanda(sessions=session,
                  conditions=conditions,
                  neural_attr='Neural Data',
                  verbose=True)

data, null = mydec.decode(training_fraction=0.7, cross_validations=10, nshuffles=25, plot=True)
print(data, null)

# Pseudo-populations are built by giving a list of sessions instead of just one data set

session_1 = {
    'Neural Data': np.random.rand(500, 55),
    'Position': np.random.rand(1000) * 10
}

session_2 = {
    'Neural Data': np.random.rand(800, 11),
    'Position': np.random.rand(1000) * 10
}

mydec = Decodanda(sessions=[session, session_1, session_2],
                  conditions=conditions,
                  neural_attr='Neural Data',
                  verbose=True)


# If more than one condition is specified, data are automatically balanced between dychotomies

data = {
    'Neural Data': np.random.rand(1000, 33),
    'Position': np.linspace(0, 10, 1000),
    'Context': np.random.rand(1000) > 0.2
}


conditions = {
    'Environment': {
        'Blue': lambda d: d['Context'] == 0,
        'Orange': lambda d: d['Context'] == 1
    },

    'Corner': {
        'Top': lambda d: d['Position'] > 9,
        'Bottom': lambda d: d['Position'] < 1
    },
}

mydec = Decodanda(sessions=data,
                  conditions=conditions,
                  neural_attr='Neural Data',
                  verbose=True)


# Let's use meaningful data now:
# FakeSession is built with rotated concatenated populations selective to three variables: color, letter, and number
# This gives a simple example of a population that has mixed selectivity with abstract representations

from decodanda.utilities import FakeSession

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


mydec = Decodanda(sessions=s1, conditions=conditions, verbose=True, neural_attr='raster')


# Decodanda exposes an interface to decode a dychotomy

number_dychotomy = [
    ['10', '11'],
    ['00', '01']
]

mydec.decode_dychotomy(number_dychotomy, training_fraction=0.7, cross_validations=1)



# And a convenient interface to automatically decode the "semantic" dychotomies, i.e. defined by each of the conditions

mydec.semantic_decode(training_fraction=0.7, plot=True, cross_validations=10, nshuffles=100)



# Similar interfaces are exposed for cross-condition performance

mydec.CCGP_dychotomy(number_dychotomy, ntrials=1, only_semantic=True)

mydec.semantic_CCGP(ntrials=1, nshuffles=100, plot=True)


# And finally, the grand geometry analysis routine

mydec.geometry_analysis(training_fraction=0.7, nshuffles=25, plot=True)


# Let's compare the geometry to a high-dimensional version of the session

s2 = FakeSession(n_neurons=150,
                 ndata=3000,
                 noise_amplitude=0.05,
                 coding_fraction=0.3,
                 rotate=True,
                 symplex=True)

mydec2 = Decodanda(sessions=s2, conditions=conditions, verbose=True, neural_attr='raster')
mydec2.geometry_analysis(training_fraction=0.7, nshuffles=25, plot=True)

