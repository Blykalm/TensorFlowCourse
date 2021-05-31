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
#----------

