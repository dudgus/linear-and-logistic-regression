import tensorflow as tf
import numpy as np

session = tf.Session()

# https://www.tensorflow.org/api_docs/python/tf/placeholder
data_values = tf.placeholder(tf.float32, shape=(4, 4))  # Shape i a matrix of 4x4 size
data_operation = tf.matmul(data_values, data_values)

# print(session.run(data_operation))  # Error because data_operation is not feeded

# Feed Random Values
matrix = np.random.rand(4, 4)
result_operation = session.run(data_operation, feed_dict={data_values: matrix})
# Same Result
result_operation_synthax = session.run(data_operation, {data_values: matrix})
print(result_operation)
print(result_operation_synthax)

# Feed Incorrect Shape Values
# matrix2 = np.random.rand(4, 5)  # Error diferent shape
# result_operation2 = session.run(data_operation, feed_dict={data_values: matrix2})

# Feed Own Values
matrix3 = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
]
result_operation3 = session.run(data_operation, feed_dict={data_values: matrix3})
print(result_operation3)


