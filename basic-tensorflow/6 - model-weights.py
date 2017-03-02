# A lot of variations here no fear is only a linear model
# Model = hypothesis = H(x) =  W*x + b = theta_1*x + theta_0
# Cost = cost(W,b)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Input
x = [1., 2., 3.]
y = [1., 2., 3.]

# Number of samples
m = n_samples = len(x)

# Set model weights
W = tf.Variable(0.0, dtype=tf.float32)

# Linear Model
hypothesis = tf.multiply(x, W)

# Cost or Error
cost = (1/m) * tf.reduce_sum(tf.pow(hypothesis - y, 2))
learn_rate = 0.1 # alpha or learn rate
optimizer = tf.train.GradientDescentOptimizer(learn_rate)
train = optimizer.minimize(cost)

# Start all variables after execute nodes
init = tf.global_variables_initializer()

# Launch the graph
session = tf.Session()
session.run(init)

# Set model weights
w0p = W.assign(5.0)
session.run(w0p)
for i in range(10):
        print(i, session.run(W))
        session.run(train)

