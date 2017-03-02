# Everything is a operation!
import tensorflow as tf

session = tf.Session()

a = tf.constant(5.0)
b = tf.constant(2.0)

c = a**b
print(c) # Tensor("pow:0", shape=(), dtype=float32)

print(session.run(c)) # 25
