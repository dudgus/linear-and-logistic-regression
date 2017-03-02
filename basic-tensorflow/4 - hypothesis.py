# A lot of variations here no fear is only a linear model
# Model = hypothesis = H(x) =  W*x + b = theta_1*x + theta_0
# Cost = cost(W,b)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# The Data - Load From File or Set
data_x = [0.0, 1.0, 2.0, 3.0, 4.0]
data_y = [0.0, 1.0, 4.0, 9.0, 16.0]  # A basic y = x**2

# print the points data
plt.scatter(data_x, data_y, color='blue')
plt.title('Input Data')
plt.xlabel('Área')
plt.ylabel('Precio')
plt.show()

# Parameters of the model
W = tf.Variable(0.0, dtype=tf.float32)
b = tf.Variable(0.0, dtype=tf.float32)
c = tf.Variable(0.0, dtype=tf.float32)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Model
hypothesis = W*x*x + b*x + c

# Cost
cost = tf.reduce_mean(tf.square(hypothesis - y))

# Minimize
alpha = tf.Variable(0.001)  # alpha = Tasa de aprendizaje
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(cost)

# Start all variables after execute nodes
init = tf.initialize_all_variables()

# Launch the graph
session = tf.Session()
session.run(init)

for i in range(10000):
        session.run(train, {x: data_x, y: data_y})
        if i % 20 == 0:
            # Line description: iteration, error, computed values
            print(i, session.run(cost, {x: data_x, y:data_y}), session.run(W), session.run(b), session.run(c))


# Plot bot values
# Generate points for the model
model_x = np.linspace(0, 4, 10)
model_y = session.run(hypothesis, {x: model_x})

plt.scatter(data_x, data_y, color='blue')
plt.plot(model_x, model_y, color='green')
plt.title('Results: Model & Input Data')
plt.xlabel('Área')
plt.ylabel('Precio')
plt.show()

# The Predictions
print('Predictions x = 10')
print(session.run(hypothesis, {x:10}))
