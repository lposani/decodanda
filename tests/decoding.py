import numpy as np
from decodanda import Decodanda, generate_synthetic_data, visualize_synthetic_data

# Test disentangling confounds: the "action" variable should be decodable when analyzed
# individually but NOT be decodable when properly balanced with the "stimulus" one.

np.random.seed(0)

data = generate_synthetic_data(n_neurons=80, n_trials=100, rateB=0, rateA=0.3, keyA='stimulus', keyB='action', corrAB=0.8)
dec = Decodanda(data=data, conditions={'action': [-1, 1], 'stimulus': [-1, 1]}, verbose=True)
dec.decode_with_nullmodel([['01'], ['10']], 0.75, nshuffles=3)


# This analysis should give a positive performance
res_unbalanced = Decodanda(data=data, conditions={'action': [-1, 1]}).decode_dichotomy('action', 0.75)

# This analysis should give a negative performance
res_balanced = Decodanda(data=data,
                         conditions={'action': [-1, 1], 'stimulus': [-1, 1]}
                         ).decode_dichotomy('action', 0.75)

# weird stuff
