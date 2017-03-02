import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Input
x = [1., 2., 3.]
y = [1., 2., 3.]

# Number of samples
m = n_samples = len(x)

# model weights
W = tf.placeholder(tf.float32)

# Model
hypothesis = tf.multiply(x, W)

# Cost or Error
cost = (1/m) * tf.reduce_sum(tf.pow(hypothesis - y, 2))

# Initializing variables
init = tf.global_variables_initializer()

# For Graphs
W_val = []
cost_val = []

# Launch the graph
session = tf.Session()
session.run(init)

for i in range(-30, 50):
    print(i * 0.1, session.run(cost, {W: i*0.1}))
    W_val.append(i*0.1)
    cost_val.append(session.run(cost, {W: i*0.1}))

# Graphs
plt.plot(W_val, cost_val, 'ro')
plt.xlabel('W')
plt.ylabel('Cost')
plt.show()
