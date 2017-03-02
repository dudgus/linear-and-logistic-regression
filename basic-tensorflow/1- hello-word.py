import tensorflow as tf

# If you comment a session NameError: name 'session' is not defined
session = tf.Session()

# Make a constant
string = tf.constant('Hello TensorFlow')
print(session.run(string))

# Try to set to other value
# string = 'Changed Value' # Error
string = tf.constant('Changed Value') # Ok
print(session.run(string))

