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
- 

  > for the future
