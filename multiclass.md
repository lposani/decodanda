Good, this is a nice contained extension – you don’t have to blow up the whole API, but you *do* have to be very deliberate about where “binary only” is baked in.

Below is a concrete checklist of what to change / add, in roughly the order I’d tackle it.

---

## 1. Relax the binary-only constraint at construction

Right now you hard-reject non-binary conditions in `__init__`:

````python
for key in conditions:
    if len(conditions[key]) != 2:
        raise RuntimeError(
            f"\n[Decodanda] In this version of Decodanda, variables should be binary\n "
            f"Variable {key} has {len(conditions[key])} values. Please check the conditions dictionary.")
``` :contentReference[oaicite:0]{index=0}  

**Step 1.1 – Remove / relax this check**

- Change it so that:
  - If all `len(conditions[key]) == 2`, you’re in “binary mode” → keep current behavior.
  - If any `len(conditions[key]) > 2`, mark the object as “multiclass capable” (e.g. `self._is_multiclass = True`) and *do not* raise.

**Step 1.2 – Decide what you want `conditions` to look like internally**

You currently funnel “discrete dict conditions” into `_generate_binary_conditions`:

```python
if type(list(conditions.values())[0]) == list:
    conditions = _generate_binary_conditions(conditions)
``` :contentReference[oaicite:1]{index=1}  

For multi-class you basically have two options:

- **Option A (cleaner for multiclass)**: keep `conditions` as “semantic”:
  - `conditions = {'stimulus': ['A', 'B', 'C'], 'action': ['left', 'right', 'none']}`
  - Introduce a new internal representation (see next section).
- **Option B (minimal changes)**: still expand to a binary code, but that becomes messy/opaque for K>2 values and the notion of “semantic dichotomies” becomes more confusing.

I would strongly lean towards **Option A** for the multiclass extension and keep `_generate_binary_conditions` only for backward compatibility / old style usage.

---

## 2. Generalize the internal representation of conditions

Right now you rely on binary words everywhere:

```python
self._condition_vectors = generate_binary_words(self.n_conditions)  # TODO change this for multimodal
self._semantic_keys = list(self.conditions.keys())
self._semantic_vectors = {string_bool(w): [] for w in generate_binary_words(self.n_conditions)}
self._generate_semantic_vectors()
``` :contentReference[oaicite:2]{index=2}  

These assume:

- One bit per variable.
- Exactly 2 values per variable.
- Condition keys like `'01'`, `'10'`, etc.

For multi-class you need to encode the full **cartesian product** of values for each variable.

**Step 2.1 – Introduce a generalized “condition vector” generator**

- New utility, e.g. `generate_condition_vectors(conditions)` that returns a list of integer vectors:

  - Suppose:
    - `conditions['stimulus'] = ['A', 'B', 'C']` → indices 0,1,2
    - `conditions['action']   = ['left', 'right']` → indices 0,1
  - Then `generate_condition_vectors` returns:
    - `[ [0,0], [0,1], [1,0], [1,1], [2,0], [2,1] ]`  

- Store as:

  ```python
  self._semantic_keys = list(self.conditions.keys())
  self._condition_vectors = <the list of int vectors>
````

**Step 2.2 – Define a canonical “condition key” from these vectors**

You currently key everything with `string_bool(w)`. For multi-class, you want something like:

* `condition_vec = [stimulus_idx, action_idx, ...]`
* Canonical key string: `'0|1'` or `'0-1-2'` or `tuple(condition_vec)` (converted to string for dict keys).

So add e.g.:

```python
def _vec_to_key(self, vec: np.ndarray) -> str:
    return '|'.join(map(str, vec))
```

Then:

```python
self._semantic_vectors = {self._vec_to_key(w): [] for w in self._condition_vectors}
self.conditioned_rasters = {self._vec_to_key(w): [] for w in self._condition_vectors}
self.conditioned_trial_index = {self._vec_to_key(w): [] for w in self._condition_vectors}
```

…and stop relying on `string_bool` for general operations in the multiclass path.

**Step 2.3 – Rewrite `_generate_semantic_vectors` to use the new representation**

Right now you do: 

```python
for condition_vec in self._condition_vectors:
    semantic_vector = '('
    for i, sk in enumerate(self._semantic_keys):
        semantic_values = list(self.conditions[sk])
        semantic_vector += semantic_values[condition_vec[i]] + ' '
    semantic_vector = semantic_vector[:-1] + ')'
    self._semantic_vectors[string_bool(condition_vec)] = semantic_vector
```

You’d change this to use `_vec_to_key` and your possibly multi-valued `conditions[sk]`.

---

## 3. Keep all the existing *binary* machinery intact (for dichotomies, CCGP, etc.)

All the fun stuff – `_powerchotomies`, `_dic_key`, `_dichotomy_from_key`, `_shuffle_conditioned_arrays`, semantic distance via `hamming`, etc. – is inherently binary and uses `string_bool` internally.  

You don’t need to rewrite that now. I’d:

* Gate them with e.g. `assert not self._is_multiclass` or leave documented as “only valid for binary variables”.
* **Do not** try to force multi-class into this binary dichotomy machinery. It will create more headaches than it solves.

So the plan is:

* **Binary mode**: everything works exactly as before.
* **Multiclass mode**: new API for decoding a *single variable* with K>2 values, using a different path.

---

## 4. Design the multiclass decoding API

Your current decoding is built around **dichotomies** and two-class training/testing:

* `_train` expects `training_raster_A`, `training_raster_B`, `label_A`, `label_B` and builds labels of two categories only. 
* `_test` does the same. 
* `_one_cv_step` builds two pooled arrays from sets of conditions `dic[0]` and `dic[1]`. 

For multi-class you want:

* Flexible number of classes K≥2.
* One **target variable** `var` with values `v_1, …, v_K`.
* All other variables treated as **nuisance** to be balanced across classes.

**Step 4.1 – Group condition keys by class of the target variable**

Given:

* `self._semantic_keys` = list of variable names in `conditions` (e.g. `['stimulus', 'action']`).
* `self._condition_vectors[i]` = `[stimulus_idx, action_idx, ...]`.
* The target variable `var` (e.g. `'stimulus'`).

You can build:

```python
# map from var -> index in condition_vec
var_index = self._semantic_keys.index(var)
values = list(self.conditions[var])  # ['A','B','C'] or [0,1,2]...

class_to_keys = {v: [] for v in values}

for vec in self._condition_vectors:
    key = self._vec_to_key(vec)
    v_idx = vec[var_index]
    v_value = values[v_idx]
    class_to_keys[v_value].append(key)
```

Each class is now a **set of condition keys** that share the same value of `var`.

This reproduces, in multiclass form, what `decode_dichotomy('stimulus')` does with two sets of binary conditions.

**Step 4.2 – Implement a new `decode_variable_multiclass` method**

Something like:

```python
def decode_multiclass(self, var, training_fraction, cross_validations=10, ndata=None, shuffled=False, return_confusion=False, subsample=0, **kwargs):
    ...
```

Per CV fold:

1. Optionally generate a neuron subset (`self._generate_random_subset`) as you do now.

2. For each class/value `c`:

   * For each underlying condition `key` in `class_to_keys[c]`:

     * Use `sample_training_testing_from_rasters(self.conditioned_rasters[key], ndata_per_condition, training_fraction, self.conditioned_trial_index[key], ...)` just like you do now in `_one_cv_step`. 
   * Concatenate all these arrays into a `training_array_c` and `testing_array_c`.

3. Stack all classes:

   ```python
   training_raster = np.vstack([training_array_c1, training_array_c2, ..., training_array_cK])
   training_labels = np.hstack([
       np.repeat(v1, training_array_c1.shape[0]),
       np.repeat(v2, training_array_c2.shape[0]),
       ...
   ])
   ```

4. Same for testing.

5. Z-scoring logic: use the same pattern you already use in `_one_cv_step` but generalized:

   * Build `big_raster = np.vstack(all training class arrays)` and z-score all training and testing data accordingly.

6. Fit classifier and compute accuracy:

   ```python
   clf = sklearn.base.clone(self.classifier)
   clf.fit(training_raster[:, self.subset], training_labels)
   preds = clf.predict(testing_raster[:, self.subset])
   acc = np.mean(preds == testing_labels)
   ```

You can either:

* Implement generic `_train_multiclass`, `_test_multiclass`, or
* Make `_train` and `_test` accept **arbitrary** label arrays and reuse them for both binary and multi-class (that is slightly nicer but requires refactoring).

**Balancing guarantee you care about**

Because you’re still sampling **per full condition** and then aggregating within classes:

* Each class’s samples are composed of equal `ndata_per_condition` from every combination of nuisance variables.
* Therefore, for each nuisance variable value (e.g. all 3 values of `var2`), you contribute the same number of samples to each `var1` class → exactly the balancing you want.

---

## 5. Multiclass confusion matrix

Once you have per-trial predictions and ground truth, confusion matrices are trivial:

* Use `from sklearn.metrics import confusion_matrix`.

Modify `decode_multiclass` to optionally return both `perfs` and the confusion matrices for each CV, or a mean confusion matrix:

* For each CV:

  ```python
  cm = confusion_matrix(testing_labels, preds, labels=values)
  ```

* Store them in a list, then either:

  * Return as `cm_list`, or
  * Average: `cm_mean = np.mean(np.stack(cm_list, axis=0), axis=0)`.

You can then add a small helper in `visualize.py`:

```python
def plot_confusion_matrix(cm, labels, ax=None, normalize=True, **imshow_kwargs):
    ...
```

And call that from a high-level wrapper, e.g.:

```python
perfs, cm = dec.decode_multiclass('stimulus', ..., return_confusion=True)
plot_confusion_matrix(cm, labels=conditions['stimulus'])
```

---

## 6. Null model behavior & shuffles for multiclass

Your current null model for dichotomies is implemented by `_shuffle_conditioned_arrays`, which swaps trials between conditions at Hamming distance 1 to preserve balancing of the other variables. 

For multiclass, you have two reasonable choices:

1. **Simple label-shuffle null**:

   * In each CV, after sampling training/testing, randomly permute the labels within each class or within each condition but preserve counts. This is simpler and robust.
2. **Extend `_shuffle_conditioned_arrays` to multiclass**:

   * Identify condition pairs that differ only in the target variable and swap like you do now.
   * This preserves more of your current “geometric null” intuition, but is more work.

Given that multiclass is a new feature, I would **start with label shuffles** for the null and, if you care, later add a more geometric version.

---

## 7. Integration with the existing `Decodanda.decode(...)` utility

You have a convenience function `Decodanda.decode(data, conditions, **decodanda_params).decode(**analysis_params)` at the bottom. 

You’ll want to decide:

* Do you add a new analysis parameter like `mode='dichotomy' | 'multiclass'` and `target_variable='stimulus'`?
* Or do you keep all “multiclass” stuff only as explicit methods (`decode_multiclass`, `decode_multiclass_null`, etc.)?

Simplest:

* Leave `decode_dichotomy` and `CCGP_dichotomy` untouched.
* Add one clean new method (`decode_multiclass`) and, optionally, a wrapper at the module level that calls it.

---

## 8. Tests / sanity checks you should run

When you’ve implemented this, I’d test the following before you send it back to me:

1. **Pure binary case**:

   * For `len(conditions[key]) == 2` for all keys, confirm:

     * `decode_dichotomy` results exactly match the previous version (within noise).
     * `balanced_resample` behavior unchanged.

2. **Toy 2×3 case, explicit balancing**:

   * Build synthetic data where `var1` has 2 values, `var2` has 3 values, and no confounds.
   * Run `decode_multiclass('var1', ...)` and verify that:

     * The per-class counts of each `var2` level in the *sampled* training and testing sets are equal (or differ only by ±1 due to integer rounding).
   * Do the same for decoding `var2` and check balance across `var1`.

3. **Confusion matrix sanity**:

   * Simulate a classifier that’s perfect or near chance and verify the confusion matrix shape and normalization.

4. **Null model**:

   * Confirm that the null model gives performance ~ chance and confusion matrices ~ uniform.

---

If you implement along these steps, you’ll end up with:

* All the original binary/dichotomy/CCGP machinery untouched.
* A clean multiclass path where the key property – **balancing nuisance variables across classes** – is preserved exactly by design.

Once you’ve hacked it in, paste me the relevant new methods (`generate_condition_vectors`, `_vec_to_key`, `decode_multiclass`, and any changes to `__init__` and `_divide_data_into_conditions`), and I can sanity-check the logic and edge cases.
