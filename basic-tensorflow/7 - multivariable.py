# Own Implementation of http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex3/ex3.html
import tensorflow as tf
import pandas as pd
import numpy as np

import sys
# 3D Plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  #Required by 3d Graphs

# The Data - Load From File
data_x = pd.DataFrame.from_csv('7 - ex3x.dat', header=None)
data_y = pd.DataFrame.from_csv('7 - ex3y.dat', header=None)
# print( data_x.head() )
# print( data_x[1] )

# Normalize data with z-score - If data no normalized not converging
data_x1_normalized = (data_x[1] - data_x[1].mean()) / data_x[1].std()
data_x2_normalized = (data_x[2] - data_x[2].mean()) / data_x[2].std()
data_y_normalized = (data_y[1] - data_y[1].mean()) / data_y[1].std()
# print( data_y_normalized )

# Print the 3D input data points
mpl.rcParams['legend.fontsize'] = 10
figure = plt.figure()
ax = figure.gca(projection='3d')
ax.scatter(data_x1_normalized, data_x2_normalized, data_y_normalized)
plt.title('Input Data - Normalized')
plt.show()

# Definir variables de TensorFlow
W1 = tf.Variable(0.0, dtype=tf.float32)
W2 = tf.Variable(0.0, dtype=tf.float32)
b = tf.Variable(0.0, dtype=tf.float32)

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Hypothesis
hypothesis = W1*x1 + W2*x2 + b

# Cost
cost = tf.reduce_mean(tf.square(hypothesis - y))

# Minimize
alpha = tf.Variable(0.01)  # alpha = Tasa de aprendizaje
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(cost)

# Start all variables after execute nodes
init = tf.global_variables_initializer()

# Launch the graph
session = tf.Session()
session.run(init)

for i in range(10000):
    session.run(train, {x1: data_x1_normalized, x2: data_x2_normalized, y: data_y_normalized})
    if i % 100 == 0:
        # Line description: iteration, error, computed values
        print(i, session.run(cost, {x1: data_x1_normalized, x2: data_x2_normalized, y: data_y_normalized}),
              session.run(W1), session.run(W2), session.run(b))


# Plot Prediction line & Normalized data
# Generate points for the model
model_x1 = np.linspace(-4, 4, 100)
model_x2 = np.linspace(-4, 4, 100)
model_y = session.run(hypothesis, {x1: model_x1, x2: model_x2})

# Print the 3D points data
figure = plt.figure()
ax = figure.gca(projection='3d')
ax.scatter(data_x1_normalized, data_x2_normalized, data_y_normalized, color='blue')
plt.plot(model_x1, model_x2, model_y, color='green')
plt.title('Results: Model & Input Data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# The Predictions
print('Predictions x1 = 1650 & x2 = 3') # Expected 399900
prediction_x1 = 1650
prediction_x2 = 3

prediction_x1_normalized = (prediction_x1 - data_x[1].mean()) / data_x[1].std()
prediction_x2_normalized = (prediction_x2 - data_x[2].mean()) / data_x[2].std()
# print('prediction_x1_normalized')
# print(prediction_x1_normalized)

prediction_y_normalized = session.run(hypothesis, {x1: prediction_x1_normalized, x2: prediction_x2_normalized})
prediction_y = (prediction_y_normalized * data_y[1].std()) + data_y[1].mean()
print(prediction_y)

# sys.exit()
