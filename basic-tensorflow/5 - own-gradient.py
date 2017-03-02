import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Input
x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

W = tf.Variable(0.0, tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

hypothesis = W*x

# Cost or Error
cost = tf.reduce_mean(tf.square(hypothesis - y))

# Minimize
learn_rate = 0.1
descent = W - tf.multiply(learn_rate, tf.reduce_mean(tf.multiply((tf.multiply(W, x)-y), x)))
update = W.assign(descent)

# Initializing variables
init = tf.global_variables_initializer()

# Launch the graph
session = tf.Session()
session.run(init)

for i in range(20):
    session.run(update, {x: x_data, y:y_data})
    # Line description: iteration, error, computed values
    print(i, session.run(cost, {x: x_data, y: y_data}), session.run(W))

# Plot Results
# Generate points for the model
model_x = np.linspace(0, 4, 20)
model_y = session.run(hypothesis, {x: model_x})

plt.scatter(x_data, y_data, color='blue')
plt.plot(model_x, model_y, color='green')
plt.title('Results: Model & Input Data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Predictions
print(session.run(hypothesis, {x: 5}))
print(session.run(hypothesis, {x: 2.5}))


