# Deep Learning

## Neural Networks

### steps

- input > weight > hidden layer 1 (activation function) > weights > hidden layer 2 (activation function) > weights > output layer

- compare output to intended output > cost function(cross entropy)

- optimization function(optimizer) > minimize cost(AdamOptimizer....SGD, AdaGrad)

- backpropagation

- feed forward + backprop = epoch

### Creat Networks Model

- imagenet -- image data
- one_hot:

  ```python
  mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # 10 classes, 0-9
    # the follow is just one_hot
    #
    # 0 = [1,0,0,0,0,0,0,0,0,0]
    # 1 = [0,1,0,0,0,0,0,0,0,0]
    # 2 = [0,0,1,0,0,0,0,0,0,0]
    # ........
  ```

- batch_size : means how big that one batch of the data fits the Networks

- `tf.placeholder()` : param : [None,784] : height x width (784 = 28 * 28)

- weights : `tf.Variable(tf.random_normal([784, n_nodes_hl1]))`

- biases : `tf.Variable(tf.random_normal(n_nodes_hl1))`

- `tf.add(tf.matmul(data, hidden_1_layer['weights']) + hidden_1_layer['biases'])`

  > (input_data * weights) +biases

- `l1 = tf.nn.relu(l1)` : Activation Function

### How the Networks Run

- `tf.nn.softmax_cross_entropy_with_logits(prediction, y)` : calculate the difference

- `tf.reduce_mean()` : Computes the mean of elements across dimensions of a tensor

- `tf.train.AdamOptimizer().minimize(cost)` : Optimizer that implements the Adam algorithm

  > learning rate = 0.001

- cycles feed forward + backprop

  > how many cycles : hm_epochs = 10

- `accuracy.eval()` : means tf.Session.run()

### Deep Learning With Our Own Data

- **nltk** --> a package for nature language processing in python

- installation : `sudo python3 -m nltk.downloader -d ~/nltk_data all`

### Simple Preprocessing language

- word_tokenize(line) : separate one line into words

- w_counts = Counter(lexicon)

  > w_counts = {'word':5454}

- words need in [50:1000] times appeared

- list += list **_vs_** list.append()

- `random.shuffle(features)` : remember to shuffle your dataset before fit

- `train_x = list(features[:, 0][:-testing_size])` : try to understand how it means

### Training and Testing our own dataset

- ```python
  i = 0
      while i < len(train_x):
          start = i
          end = i + batch_size
          batch_x = np.array(train_x[start:end])
          batch_y = np.array(train_y[start:end])

          _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                        y: batch_y})
          epoch_loss += c
          i += batch_size
  ```

- Use own data means we just change the batch dealing process

## Recurrent Neural Network

### Basics

- the traditional nn is [X -> Layer -> Output]

- This is the [RNN](http://blog.csdn.net/heyongluoyao8/article/details/48636251)![the RNN](http://kvitajakub.github.io/img/rnn-unrolled.svg)

- the Hidden Layer is connected

- Most used in [NLP(Natural language processing)](https://en.wikipedia.org/wiki/Natural_language_processing)

### RNN examples

- `tf.transpose(x, [1, 0, 2])` : just matrix transpose

- transpose in numpy

  ```python
  import numpy as np

  x = np.ones((1,2,3))

  print(x)
  print(np.transpose(x,(1,0,2)))

  # [[
  #   [ 1\.  1\.  1.],
  #   [ 1\.  1\.  1.]
  # ]]
  #
  # [
  #  [[ 1\.  1\.  1.]],
  #  [[ 1\.  1\.  1.]]
  # ]
  ```

- tensorflow needs to transpose into (1,0,2) means 3-d -> 3-d ; 1-d -> 2-d ; 2-d -> 1-d

## Convolutional Neural Network

### Basics

- Convolution + Pool = Hidden Layer

- Fully Connected : just the same as the ordinary nn

- the whole structure: ![](http://cs231n.github.io/assets/nn1/neural_net2.jpeg)

- the hidden layer: ![](http://cs231n.github.io/assets/cnn/cnn.jpeg)

- [convolutional-networks](http://cs231n.github.io/convolutional-networks/#overview)

### CNN examples

- `weights : tf.Variable(tf.random_normal([5,5,1,32])` : 5 * 5 convolution ,1 input ,32 outputs
- `weights : W_fc':tf.Variable(tf.random_normal([7*7*64,1024]))` : fully connected 7*7 means we just need the part of the images
- `tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')`

  - A list of ints. The stride of the sliding window for each dimension of input. Must be in the same order as the dimension specified with format
  - padding: A string from: "SAME", "VALID". The type of padding algorithm to use

- `tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')`

  - ksize: The size of the window for each dimension of the input tensor.
  - strides: The stride of the sliding window for each dimension of the input tensor

- `fc = tf.nn.dropout(fc, keep_rate)`

  - x: A tensor
  - keep_prob: A scalar Tensor with the same type as x. The probability that each element is kept

## TFLearn

> [tflearn.org](https://tflearn.org)
