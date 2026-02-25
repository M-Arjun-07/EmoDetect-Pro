import tensorflow as tf

print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Force simple GPU computation
with tf.device('/GPU:0'):
    a = tf.random.normal([1000, 1000])
    b = tf.matmul(a, a)
    print("Matrix multiplication done on GPU")

print("Test completed successfully.")