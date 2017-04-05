# Notes in Machine Learning with Python

--------------------------------------------------------------------------------

## Environment Problem

- [numpy and scipy](https://www.zhihu.com/question/30188492?sort=created "problem on windows")
- normal ones:pip install pandas quandl sklearn numpy matplotlib
- [pythonprogramming.net](https://pythonprogramming.net/ "Great")
- [github](https://github.com/ProHiryu/Coursera_courses/tree/master/courses/Machine%20Learning%20with%20Python)

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

## Kernels

> change the linearly inseparable data into a linearly separable data

### Why Kernels

```
K(x ,x') = z.z' //The Kernel function

z = function(x)
z' = function(x-)

X = [x1,x2] //simple dataset

// the kernel wants to be 2nd order polynomial

Z = [1,x1,x2,x1^2,x2^2,x1x2]

Z' = [1,x1',x2',x1'^2,x2'^2,x1'x2']

K(x ,x') = Z.Z' = 1+x1x1'+ ... +x1x2x1'x2'

K(x, x') = (1 + x1x1' + ... + xnxn')^p // polynomial Kernel

p: order of polynomial
n: dimensions

// Learn more about Exponential kernel for RBF
```

### Soft Margin SVM

> google: svm kernel visualization overlapping --> softmax polynomial kernel
