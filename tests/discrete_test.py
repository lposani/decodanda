from decodanda import *
from decodanda.utilities import FakeSession


def main():
    # Generate dummy data with abstract representation of three variables: letter, color, number

    s1 = FakeSession(100, 2500, noise_amplitude=0.1, coding_fraction=0.25, rotate=True)
    s2 = FakeSession(50, 5000, noise_amplitude=0.1, coding_fraction=0.25, rotate=True)
    s3 = FakeSession(75, 1550, noise_amplitude=0.1, coding_fraction=0.25, rotate=True)

    # Put the data in a dictionary format

    dict1 = {'raster': s1.raster, 'behaviour_letter': s1.behaviour_letter, 'behaviour_number': s1.behaviour_number, 'behaviour_color': s1.behaviour_color}
    dict2 = {'raster': s2.raster, 'behaviour_letter': s2.behaviour_letter, 'behaviour_number': s2.behaviour_number, 'behaviour_color': s2.behaviour_color}
    dict3 = {'raster': s3.raster, 'behaviour_letter': s3.behaviour_letter, 'behaviour_number': s3.behaviour_number, 'behaviour_color': s3.behaviour_color}

    all_the_sessions = [dict1, dict2, dict3]

    # Define the conditions we want to decode

    conditions = {
        'behaviour_letter': ['A', 'B'],
        'behaviour_number': [1, 2],
        'behaviour_color': ['green', 'red']
    }


    # Use DictDecodanda for decoding pseudo-simultaneous data

    dec = DictDecodanda(all_the_sessions, conditions=conditions, verbose=True,
                        exclude_silent=False, neural_attr='raster')

    performances = dec.decode(training_fraction=0.5, ntrials=10, plot=True)
    ccgp, ccgp_null = dec.CCGP(ntrials=1, nshuffles=10, plot=True)
    plt.show()


if __name__ == '__main__':
    main()








