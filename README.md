```Decodanda``` is a best-practices-made-easy Python package for population activity decoding in neuroscience. 

Some of the best practices handled by the package include:
- Balancing classes
- Cross validation
- Creation of pseudo-population data
- Null model to test significance of the performance
- When handling multiple variables, ```Decodanda``` balances the data to disentangle the individual variables and avoid the confounding effects of correlated conditions.

Please refer to ```examples.ipynb``` for some usage examples.

For a guided explanation of some of the best practices implemented by Decodanda, you can refer to my teaching material for the Advanced Theory Course in Neuroscience at Columbia University: https://tinyurl.com/ArtDecod

For any feedback, please contact me through this module: https://forms.gle/iifpsAAPuRBbYzxJ6


Have fun!


## Getting started

All decoding functions are implemented as methods of the ```Decodanda``` class. 
The constructor of this class takes two main objects:

- ```sessions```: a single dictionary, or a list of dictionaries, containing the data to analyze. 
In the case of N neurons and T trials (or time bins), each session must contain:
  - the **neural data**: 
  
    ```TxN``` array, under the ```raster``` key (you can specify a different one)
  - the **values of the variables** we want to decode

    ```Tx1``` array per variable
  - the **trial** number (independent samples for cross validation): 
     
    ```Tx1``` array


- ```conditions```: a dictionary specifying what values we want to decode for each variable in session.

For example, if we want to decode the variable ```letter```, which takes values ```A, B``` from simultaneous recordings of N neurons x T trials we will have:
```
session = {
    'raster': [[0, 1, ..., 0], ..., [0, 0, ..., 1]]   # <TxN array>, neural activations 
    'letter': ['A', 'A', 'B', ..., 'B']               # <Tx1 array>, labels
    'trial':  [1, 2, 3, ..., T]                       # <Tx1 array>, trial number
}

conditions = {
    'letter': ['A', 'B']
}

my_decodanda = Decodanda(
                sessions=session,
                conditions=conditions,
                trial_attr='trial')

``` 

To start the decoding analysis, just call the ```decode()``` method

```
performances, null = my_decodanda.decode(
                        training_fraction=0.5,  # fraction of trials used for training
                        cross_validations=10,   # number of cross validation folds
                        nshuffles=20)           # number of null model iterations
                                        
>>> performances
{'letter': 0.84}
>>> null
{'letter'" [0.55, 0.43, 0.57 ... 0.52]}  # 20 values
```