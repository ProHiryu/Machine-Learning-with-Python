# Clustering and Unsupervised Machine Learning

## dealing with non-numeric data

- get the unique value of the columns
- set the numbers of value id
- use the id as the numeric data of this column

  ```python
  def handle_non_numerical_data(df):
        columns = df.columns.values
        for column in columns:
            text_digit_vals = {}
            def convert_to_int(val):
                return text_digit_vals[val]

            if df[column].dtype != np.int64 and df[column].dtype != np.float64:
                column_contents = df[column].values.tolist()
                unique_elements = set(column_contents)
                x = 0
                for unique in unique_elements:
                    if unique not in text_digit_vals:
                        text_digit_vals[unique] = x
                        x+=1

                df[column] = list(map(convert_to_int, df[column]))

        return df
  ```

- `set()` --> get the unique values

- `map()` --> apply the function for element in the list

## K - Means

- **`X = preprocessing.scale(X)` is necessarily!**

- `df.drop(['ticket'], 1, inplace = True)` Do not forget the **Parameters**

  > the '1' and 'inplace = True'

- [np.linalg.norm()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html) --> calculate the distance between two points, there is many options , use it well

- [np.average](https://docs.scipy.org/doc/numpy/reference/generated/numpy.average.html) --> [axis](https://docs.scipy.org/doc/numpy/glossary.html) = 0 means A 2-dimensional array has two corresponding axes: the first running vertically downwards across rows (axis 0), and the second running horizontally across columns (axis 1).

- An 3d figure see [mplot3d](http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html)

  ```python
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D

  ...

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ```

## Mean - Shift

- initialize one point as centroid, set bandwidth and class order

- find all points in the circle (centroid, bandwidth), and set the possibility of class(point) += 1

- calculate the vector of all the points in class, shift = sum(vectors)

- center = center + shift

- replay the 2-4 step, make a convergence ![mean - shift](http://pic002.cnblogs.com/images/2012/358029/2012051215101233.jpg)

- if the centroid is in the circle of an existed centroid, merge the two class, otherwise make a new class

- replay 1-5 until all the points are marked

- classify: find index of max(class(point)) set index as the class

- `print pd.df.describe()` --> show more simply one
