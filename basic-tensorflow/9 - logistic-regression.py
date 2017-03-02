import tensorflow as tf
import numpy as np


import sys
import matplotlib.pyplot as plt


# The Data - Load From File
data = np.loadtxt('9 - train.dat', unpack=True, dtype='float32')
print('Input Data')
print(data)

data_x = data[0:-1]
data_y = data[-1]
print('data_x')
print(data_x)

print('data_y')
print(data_y)


# Print scatter the input data points
red_points = np.where(data_y == 1)
black_points = np.where(data_y == 0)
print('red_points')
print(red_points)

plot_red_x = data_x[1, red_points]
plot_red_y = data_x[2, red_points]
print('plot_red_x')
print(plot_red_x)

plot_black_x = data_x[1, black_points]
plot_black_y = data_x[2, black_points]

plt.scatter(x=plot_red_x, y=plot_red_y, color='red', marker='o')
plt.scatter(x=plot_black_x, y=plot_black_y, color='black', marker='x')
plt.show()

# Definir variables de TensorFlow
W = tf.Variable(tf.random_uniform([1, len(data_x)], -1.0, 1.0))
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Hypothesis
h = tf.matmul(W, x)
hypothesis = tf.div(1.0, 1.0 + tf.exp(-h))

# Cost
cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))

# Minimize
alpha = tf.Variable(0.1)  # alpha = Tasa de aprendizaje
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(cost)

# Start all variables after execute nodes
init = tf.global_variables_initializer()

# Launch the graph
session = tf.Session()
session.run(init)

for i in range(10000):
    session.run(train, {x: data_x, y: data_y})
    if i % 1000 == 0:
        # Line description: iteration, error, computed values
        print(i, session.run(cost, {x: data_x, y: data_y}),
              session.run(W))

# Predictions
print('Predictions')
session.run(hypothesis, {x: [[1], [2], [2]]})

sys.exit()

# Plot the results
model_x = np.linspace(1, 8, 100)
model_y = session.run(hypothesis, {x: model_x})

plt.scatter(x=plot_red_x, y=plot_red_y, color='red', marker='o')
plt.scatter(x=plot_black_x, y=plot_black_y, color='black', marker='x')
plt.plot(x=model_x, y=model_y, color='green')
plt.title('Results: Model & Input Data')
plt.show()

