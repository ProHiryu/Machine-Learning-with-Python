# K Nearest Neighbors Application

- [dataset](http://archive.ics.uci.edu/ml/datasets.html "UCI")
- numpy.reshape :

```python
example_measures = np.array([4,2,1,1,1,2,3,2,1])
example_measures = example_measures.reshape(len(example_measures),-1)

## a：将要被重塑的类数组或数组
## newshape：整数值或整数元组。新的形状应该兼容于原始形状。如果是一个整数值，表示一个一维数组的长度；
## 如果是元组，一个元素值可以为-1，此时该元素值表示为指定，此时会从数组的长度和剩余的维度中推断出
```

- lib - warnings : `warnings.warn('K is set to a value less than total voting groups!')`

- numpy.linalg.norm : `np.linalg.norm(np.array(features) - np.array(predict))`

- **python dictionary**:

  ```python
  dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
    new_features = [5,7]
    for group in dataset:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
  ```

- Lib - Counters : `from collections import Counter`

  - `vote_result = Counter(votes).most_common(1)[0][0]`
  - It gives us a list of tuple,the '1' in here determines the numbers of the most common tuples
  - tuples:(the most common element,numbers of the most common)

- use the [-num] of list flexbily

  ```python
  test_size = 0.2
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}
    train_data = full_data[:-int(test_size*len(full_data))]
    test_data = full_data[-int(test_size*len(full_data)):]

    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set[i[-1]].append(i[:-1])
  ```
