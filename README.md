# Notes in Machine Learning with Python

--------------------------------------------------------------------------------

## Environment Problem

- [numpy and scipy](https://www.zhihu.com/question/30188492?sort=created "problem on windows")
- normal ones:pip install pandas quandl sklearn numpy matplotlib
- [pythonprogramming.net](https://pythonprogramming.net/ "Great")
- [github](https://github.com/ProHiryu/Coursera_courses/tree/master/courses/Machine%20Learning%20with%20Python)

## Regression

- [reference](http://pandas.pydata.org/pandas-docs/stable/indexing.html)
- `dataframe.shift(num)` just move the dataset num times rightwards
- `dataframe.iloc[num]` Selection by Position - is primarily integer position based (from 0 to length-1 of the axis)
- `dataframe.loc[text]` Selection by Label
- python list - `list[-1]` the last element of the list
- python date:

  - time string : `time.ctime()` 'Mon Dec 17 21:02:55 2012'
  - datetime tuple(datetime obj) : `datetime.now()` `datetime.datetime(2012, 12, 17, 21, 3, 44, 139715)`
  - time tuple(time obj) : `time.struct_time(tm_year=2008, tm_mon=11, tm_mday=10, tm_hour=17, tm_min=53, tm_sec=59, tm_wday=0, tm_yday=315, tm_isdst=-1)`
  - timestamp : 时间戳类型:自1970年1月1日(00:00:00 GMT)以来的秒数
  - ![NOT FOUND!](http://s9.sinaimg.cn/large/b09d4602xd10ea8f9ab88&690 "change relation")

- **python iteration `np.nan for _ in range()` means nothing,the variable is not going to be used** - [nan, nan, nan, nan, nan, 781.29480101781269]

- python list : xs*ys means every element in xs times ys by the order of index

## Matplotlib

```python
style.use('ggplot')

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
```

- `plt.scatter()` ---> scatter

## Pickle

```python
# clf = LinearRegression(n_jobs=-1)
# clf.fit(X_train, y_train)
# with open('linearregression.pickle','wb') as f:
#     pickle.dump(clf, f)

pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)
```

## Classification

### K Nearest Neighbors Application

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

## Kernels

> change the linearly inseparable data into a linearly separable data

$$ x=\frac{-b\pm\sqrt{b^2-4ac}}{2a} $$
