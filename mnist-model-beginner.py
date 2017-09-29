# Loading MNIST data from tensorflow
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Starting ineteractive session in tensorflow
import tensorflow as tf
sess = tf.InteractiveSession()

# Place holders for input features and labels
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Defining variables for weights and Biases
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Initializing all variables defined
sess.run(tf.global_variables_initializer())

# Defining the Prediction model and Loss function
y = tf.matmul(x, W) + b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Using gradient descent to reduce the loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Running gradient descent for defined range below to reduce the loss, we'll be using Stochastic gradient descent, 
# whcih means we use part of training data to train our model in each iteration
for _ in range(1000):
    batch_x, batch_y = mnist.train.next_batch(100)
    train_step.run(feed_dict={x:batch_x, y_:batch_y})

## Evaluating the Model ##
# Detecting where the model predicted correctly and calculating accuracy
correct_predictions = tf.equal(tf.arg_max(y,1), tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

#printing the accuracy of the model on test data
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))