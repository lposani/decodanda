```Decodanda``` is a best-practices-made-easy Python package for population activity decoding in neuroscience. 

Some of the best practices handled by the package include:
- Balancing classes
- Cross validation
- Creation of pseudo-population data
- Null model to test significance of the performance
- When handling multiple variables, ```Decodanda``` balances the data to disentangle the individual variables and avoid the confounding effects of correlated conditions.

Please refer to [examples.ipynb](https://github.com/lposani/decodanda/blob/master/examples.ipynb) for some usage examples.

For a guided explanation of some of the best practices implemented by Decodanda, you can refer to [my teaching material](https://tinyurl.com/ArtDecod) for the Advanced Theory Course in Neuroscience at Columbia University.

For any feedback, please contact me through [this module](https://forms.gle/iifpsAAPuRBbYzxJ6).


Have fun!


## Getting started
### Decoding one variable from neural activity

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
```python
session = {
    'raster': [[0, 1, ..., 0], ..., [0, 0, ..., 1]],   # <TxN array>, neural activations 
    'letter': ['A', 'A', 'B', ..., 'B'],               # <Tx1 array>, labels
    'trial':  [1, 2, 3, ..., T],                       # <Tx1 array>, trial number
}

conditions = {
    'letter': ['A', 'B']
}

my_decodanda = Decodanda(
                sessions=session,
                conditions=conditions)

```

We can decode `letter` from the activity by calling the ```decode()``` method

```python
performances, null = my_decodanda.decode(
                        training_fraction=0.5,  # fraction of trials used for training
                        cross_validations=10,   # number of cross validation folds
                        nshuffles=20)           # number of null model iterations
```
which outputs
```text
>>> performances
{'letter': 0.84}  # mean decoding performance over the cross_validations folds
>>> null
{'letter' [0.55, 0.43, 0.57 ... 0.52]}  # nshuffles (20) values
```



<br/>

### Decoding multiple variables from neural activity
TODO

### Balance behavior to compare decoding performances
TODO

### Decoding from pseudo-populations data 
TODO

### CCGP 
TODO

### `Decodanda()` constructor parameters

| parameter                   | type                                                                                                                                                                      | description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|-----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `sessions`                  | `dict` or `list` of dicts                                                                                                                                                 | The data used to decode, organized in dictionaries. Each session dictionary object must contain <br/> - one or more variables and values we want to decode, each in the format <br/> `<var name>: <Tx1 array of values>` <br/> -`raster: <TxN array>`<br/> the neural features from which we want to decode the variable values <br/> - a `trial: <Tx1 array>`<br/> the number that specify which chunks of data are considered independent for cross validation <br/> <br/> if more than one sessions are passed to the constructor, `Decodanda` will create pseudo-population data by combining trials from the different sesisons. |
| `conditions`                | `dict`                                                                                                                                                                    | A dictionary that specifies which values for which variables of `sessions` we want to decode, in the form `{key: [value1, value2]}` <br/><br/>If more than one variable is specified, `Decodanda` will balance all conditions during each decoding analysis to disentangle the variables and avoid confounding correlations.                                                                                                                                                                                                                                                                                                          |
| `classifier`                | Possibly a `scikit-learn` classifier, but any object that exposes `.fit()`, `.predict()`, and `.score()` methods should work. <br/><br/> default: `sklearn.svm.LinearSVC` | The classifier used for all decoding analyses                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `neaural_attr`              | `string` <br/><br/> default: `'raster'`                                                                                                                                   | The key of the neural features in the `session` dictionary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `trial_attr`                | `string` or `None`                                                                                                                                                        | The key of the trial attribute in the `session` dictionary. Each different trial is considered as an independent sample to be used in the cross validation routine, i.e., vectors with the same trial number always goes in either the training or the testing batch. If `None`: each contiguous chunk of the same values of all variables will be considered an individual trial.                                                                                                                                                                                                                                                    |
| `trial chunk`               | `int` or `None` <br/><br/>default: `None`                                                                                                                                 | Only used when `trial_attr=None`. The maximum number of consecutive data points with the same value of all variables that are numbered with the same trial number.                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `exclude_contiguous_chunks` | `bool`<br/><br/>default: `False`                                                                                                                                          | Only used when `trial_attr=None` and `trial_chunks != None`. Discards trials, defined as chunks of `trial_chunk` data each with the same variable values, that are consecutive in time. Useful to avoid decoding temporal artifacts when there are long auto-correlation times in the neural activations (e.g., calcium imaging)                                                                                                                                                                                                                                                                                                      |
| `min_data_per_condition`    | `int`<br/><br/>default: 2                                                                                                                                                 | The minimum number of data points per each *condition*, defined as a specific combination of variable values, that a session needs to have to be included in the analysis. In the case of pseudo-simultaneous data, sessions that do not meet this criterion will be excluded from the analysis. If no sessions meet the criterion, the constructor will raise an error.                                                                                                                                                                                                                                                              |
| `min_trials_per_condition`  | `int`<br/><br/>default: 2                                                                                                                                                 | The minimum number of unique trial numbers per each *condition*, defined as a specific combination of variable values, that a session needs to have to be included in the analysis. In the case of pseudo-simultaneous data, sessions that do not meet this criterion will be excluded from the analysis. If no sessions meet the criterion, the constructor will raise an error.                                                                                                                                                                                                                                                     |
| `exclude_silent`            | `bool`<br/><br/>default: `False`                                                                                                                                          | Excludes all silent population vectors (only zeros) from the analysis.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `verbose`                   | `bool`<br/><br/>default: `False`                                                                                                                                          | If `True`, prints most operations and analysis results.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `fault_tolerance`           | `bool`<br/><br/>default: `False`                                                                                                                                          | If `True`, raises a warning instead of an error when no sessions meet the inclusion criteria.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |

<br/>
