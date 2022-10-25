from decodanda import *
import time

# We create a synthetic data set where neurons respond to two variables, labeled as
#  - behaviour_letter (taking values A, B)
#  - behaviour_number (taking values 1, 2)
# neural activity is stored under the 'raster' keyword
#
# Through the geometry_analysis() function, here we test
# - the decoding() function
# - the CCGP() function
# - all the dichotomy logic


def main(nshuffles):
    conditions = {
        'behaviour_number': [1, 2],
        'behaviour_letter': ['A', 'B']
    }

    # Disentangled representations
    s1 = FakeSession(n_neurons=120,
                     ndata=1000,
                     noise_amplitude=0.02,
                     coding_fraction=0.3,
                     rotate=False,
                     symplex=False)

    decodanda_params = {
        'verbose': False,
    }

    analysis_params = {
        'cross_validations': 10,
        'training_fraction': 0.8,
        'nshuffles': nshuffles,
        'XOR': True
    }

    t0 = time.time()
    res, null = decoding_analysis(data=s1, conditions=conditions, decodanda_params=decodanda_params,
                                  analysis_params=analysis_params, parallel=True, plot=True)
    plt.title('Parallel, nshuffles=%u' % nshuffles)

    dt_par = time.time() - t0


    decodanda_params = {
        'verbose': False,
    }

    analysis_params = {
        'cross_validations': 10,
        'training_fraction': 0.8,
        'nshuffles': nshuffles,
        'XOR': True
    }

    t0 = time.time()
    res, null = decoding_analysis(data=s1, conditions=conditions, decodanda_params=decodanda_params,
                                  analysis_params=analysis_params, parallel=False, plot=True)
    plt.title('Serial, nshuffles=%u' % nshuffles)
    dt_ser = time.time() - t0
    print('\nNshuffles =', nshuffles)
    print("Parallel time execution: %.2f s" % dt_par)
    print("Serial time execution: %.2f s" % dt_ser)


if __name__ == '__main__':
    main(4)
    main(12)
    main(64)
    main(128)
    plt.show()

