"""
in this file we are covering some of the fundamentals of tensorflow

More specifically, we're going to cover:
* Introduction to tensor
* Getting information from tensor
* Manipulating tensors
* Tensors & numpy
* Using @tf.function (a way to speed up your regular Python functions)
*Using GPU's with TensorFlow (or TPU's)
*Exercises to try for yourself
"""

# introduction to Tensors
import tensorflow as tf

print(tf.__version__)
########################################################################################################################
# create Tensors with tf.constant()
########################################################################################################################
scalar = tf.constant(7)
print(scalar)
print("_______________________________________________________________________________________________________________")

# check the number of dimensions of a tensor (ndim stands for number of dimensions)
print(scalar.ndim)
print("_______________________________________________________________________________________________________________")

# create a vector
vector = tf.constant([10, 10])
print(vector)
print("_______________________________________________________________________________________________________________")

# check the dimension of the vector
print(vector.ndim)
print("_______________________________________________________________________________________________________________")

# create a matrix (has more than 1 dimension)
matrix = tf.constant([[10, 7],
                      [7, 10]])
print(matrix)
print("_______________________________________________________________________________________________________________")

# create another matrix
another_matrix = tf.constant([[10., 7.],
                              [3., 2.],
                              [8., 9.]], dtype=tf.float16)  # specify the data tye with dtype param
print(another_matrix)
print("_______________________________________________________________________________________________________________")

# what is the number of dimensions of another_matrix
print(another_matrix.ndim)
print("_______________________________________________________________________________________________________________")

# lets create a tensor
tensor = tf.constant([[[1, 2, 3],
                       [4, 5, 6]],
                      [[7, 8, 9],
                       [10, 11, 12]],
                      [[13, 14, 15],
                       [16, 17, 18]]])
print(tensor)
print(tensor.ndim)
print("_______________________________________________________________________________________________________________")

"""
what we've learned so far:

* Scaler: a single number
* Vector: a number with direction (e.g. wind speed and direction)
* Matrix: a 2 dimensional array of numbers
* Tensor: an n-dimensional array of number ( where n can by any number, a 0-dimensional tensor is a scalar, a 1-dimensional tensor is a vector) 
"""

#######################################################################################################################
# Creating a tensor with tf.variable
#######################################################################################################################

# create the same tensor with tf.variable() as above
changeable_tensor = tf.Variable([10, 7])
unchangeable_tensor = tf.constant([10, 7])
print(changeable_tensor)
print(unchangeable_tensor)
print("_______________________________________________________________________________________________________________")

# lets try to change one of the elements in our changeable tensor using .assign (spoiler you can only use .assign to chane a value in a changeable tensor)
changeable_tensor[0].assign(7)
print(changeable_tensor)
print("_______________________________________________________________________________________________________________")

# lets try to change our unchangeable tensor ( spoiler it does not work)
# unchangeable_tensor[0].assign(7)
# print(unchangeable_tensor)
# print("_______________________________________________________________________________________________________________")

"""
Note: Rarely in practice will you need to decide whether to use tf.constant or tf.Variable to create tensors,
as TensorFlow dies this for you. However, if in doubt, use tf.constant and change it later if needed.
"""
########################################################################################################################
# Create Random tensors
########################################################################################################################

# Random tensors are tensors of some arbitrary size which contain random number.

# Create two random tensors
random_1 = tf.random.Generator.from_seed(42)  # set seed for reproducibility
random_1 = random_1.normal(shape=(3, 2))
random_2 = tf.random.Generator.from_seed(42)
random_2 = random_2.normal(shape=(3, 2))
# Are they equal?
print(random_1)
print(random_2)
print(random_1 == random_2)
print("_______________________________________________________________________________________________________________")

########################################################################################################################
# Shuffle the order of elements in a tensor
########################################################################################################################

# Shuffle a tensor (Valuable for when you want to shuffle your data so the inherit order doesn't effect learning)
not_shuffled = tf.constant([[10, 7],
                            [3, 4],
                            [2, 5]])
print(not_shuffled.ndim)
print(not_shuffled)
print("_______________________________________________________________________________________________________________")

# shuffle our non-shuffled tensor
print(tf.random.shuffle(not_shuffled))
print("_______________________________________________________________________________________________________________")

########################################################################################################################
# Other ways to make tensors
########################################################################################################################

# Create a tensor of all ones
print(tf.ones([10, 7]))
print("_______________________________________________________________________________________________________________")

# Create a tensor with all zeroes
print(tf.zeros([10, 7]))
print("_______________________________________________________________________________________________________________")

# You can also turn NumPy arrays into tensors
"""
The main difference between NumPy arrays and TensorFlow tenors is that tensors can be run on a GPU (much faster for numerical computing).
"""
import numpy as np

numpy_A = np.arange(1, 25, dtype=np.int32)  # Create a NumPy array between 1 and 25
print(numpy_A)
print("_______________________________________________________________________________________________________________")
A = tf.constant(numpy_A)
print(A)
print("_______________________________________________________________________________________________________________")
B = tf.constant(numpy_A, shape=(2, 3, 4))
print(B)
print("_______________________________________________________________________________________________________________")

# X = tf.constant(some_matrix) # capital for matrix or tensor
# y = tf.constant(vector) #non-capital for vector
