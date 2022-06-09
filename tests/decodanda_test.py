from decodanda import *
from decodanda.utilities import FakeSession

# Generate dummy data with abstract representation of three variables: letter, color, number
noise = 0.05
rotate = 1
symplex = 0

s1 = FakeSession(150, 2500, noise_amplitude=noise, coding_fraction=0.1, rotate=rotate, symplex=symplex)
s2 = FakeSession(115, 5000, noise_amplitude=noise, coding_fraction=0.1, rotate=rotate, symplex=symplex)
s3 = FakeSession(75, 1550, noise_amplitude=noise, coding_fraction=0.1, rotate=rotate, symplex=symplex)
s2.name = 'SessioneFinta2.0'
s3.name = 'SessioneFinta3.0'

# Define the functions that specify the conditions to decode

semantic_conditions = {

    'number': {
        '1': lambda s: s.behaviour_number < 1.5,
        '2': lambda s: s.behaviour_number > 1.5
    },

    'letter': {
        'A': lambda s: s.behaviour_letter == 'A',
        'B': lambda s: s.behaviour_letter == 'B'
    },

    'color': {
        'g': lambda s: s.behaviour_color == 'green',
        'r': lambda s: s.behaviour_color == 'red'
    }
}

# Use Decodanda with pseudo-simultaneous

d = Decodanda([s1, s2, s3],
              semantic_conditions,
              verbose=True,
              exclude_silent=True,
              neural_attr='raster',
              trial_chunk=50,
              exclude_contiguous_chunks=False)

d.geometry_analysis(training_fraction=0.8, nshuffles=11, cross_validations=5, plot=True,
                    savename='./tests/geometry_lowD')


# Dummy data with HIGH-dimensional representation of three variables: letter, color, number

noise = 0.05
rotate = 1
symplex = 1

s1 = FakeSession(150, 2500, noise_amplitude=noise, coding_fraction=0.1, rotate=rotate, symplex=symplex)
s2 = FakeSession(115, 5000, noise_amplitude=noise, coding_fraction=0.1, rotate=rotate, symplex=symplex)
s3 = FakeSession(75, 1550, noise_amplitude=noise, coding_fraction=0.1, rotate=rotate, symplex=symplex)
s2.name = 'SessioneFinta2.0'
s3.name = 'SessioneFinta3.0'


# Use Decodanda with pseudo-simultaneous

d = Decodanda([s1, s2, s3],
              semantic_conditions,
              verbose=True,
              exclude_silent=True,
              neural_attr='raster',
              trial_chunk=50,
              exclude_contiguous_chunks=False)

d.geometry_analysis(training_fraction=0.8, nshuffles=11, cross_validations=5, plot=True,
                    savename='./tests/geometry_highD')


plt.show()
