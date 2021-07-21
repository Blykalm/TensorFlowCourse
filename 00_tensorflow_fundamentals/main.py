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
print("_______________________________________________1_______________________________________________________________")

# check the number of dimensions of a tensor (ndim stands for number of dimensions)
print(scalar.ndim)
print("_______________________________________________2_______________________________________________________________")

# create a vector
vector = tf.constant([10, 10])
print(vector)
print("_______________________________________________3_______________________________________________________________")

# check the dimension of the vector
print(vector.ndim)
print("_______________________________________________4_______________________________________________________________")

# create a matrix (has more than 1 dimension)
matrix = tf.constant([[10, 7],
                      [7, 10]])
print(matrix)
print("_______________________________________________5_______________________________________________________________")

# create another matrix
another_matrix = tf.constant([[10., 7.],
                              [3., 2.],
                              [8., 9.]], dtype=tf.float16)  # specify the data tye with dtype param
print(another_matrix)
print("_______________________________________________6_______________________________________________________________")

# what is the number of dimensions of another_matrix
print(another_matrix.ndim)
print("_______________________________________________7_______________________________________________________________")

# lets create a tensor
tensor = tf.constant([[[1, 2, 3],
                       [4, 5, 6]],
                      [[7, 8, 9],
                       [10, 11, 12]],
                      [[13, 14, 15],
                       [16, 17, 18]]])
print(tensor)
print(tensor.ndim)
print("_______________________________________________8_______________________________________________________________")

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
print("_______________________________________________9_______________________________________________________________")

# lets try to change one of the elements in our changeable tensor using .assign (spoiler you can only use .assign to chane a value in a changeable tensor)
changeable_tensor[0].assign(7)
print(changeable_tensor)
print("_______________________________________________10______________________________________________________________")

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
print("_______________________________________________11______________________________________________________________")

########################################################################################################################
# Shuffle the order of elements in a tensor
########################################################################################################################

# Shuffle a tensor (Valuable for when you want to shuffle your data so the inherit order doesn't effect learning)
not_shuffled = tf.constant([[10, 7],
                            [3, 4],
                            [2, 5]])
print(not_shuffled.ndim)
print(not_shuffled)
print("_______________________________________________12______________________________________________________________")

# shuffle our non-shuffled tensor
print(tf.random.shuffle(not_shuffled))
print("_______________________________________________13______________________________________________________________")

########################################################################################################################
# Other ways to make tensors
########################################################################################################################

# Create a tensor of all ones
print(tf.ones([10, 7]))
print("_______________________________________________14______________________________________________________________")

# Create a tensor with all zeroes
print(tf.zeros([10, 7]))
print("_______________________________________________15______________________________________________________________")

# You can also turn NumPy arrays into tensors
"""
The main difference between NumPy arrays and TensorFlow tenors is that tensors can be run on a GPU (much faster for numerical computing).
"""
import numpy as np

numpy_A = np.arange(1, 25, dtype=np.int32)  # Create a NumPy array between 1 and 25
print(numpy_A)
print("_______________________________________________16______________________________________________________________")
A = tf.constant(numpy_A)
print(A)
print("_______________________________________________17______________________________________________________________")
B = tf.constant(numpy_A, shape=(2, 3, 4))
print(B)
print("_______________________________________________18______________________________________________________________")

# X = tf.constant(some_matrix) # capital for matrix or tensor
# y = tf.constant(vector) #non-capital for vector

########################################################################################################################
# Gather data from our tensors.
########################################################################################################################
"""
When dealing with tensors you probably want to be aware of the following attributes:
* shape
* Rank
* Axis or dimension
* Size
"""

# Create a rank 4 tensor (4 dimensions)

rank_4_tensor = tf.zeros(shape=[2, 3, 4, 5])
print(rank_4_tensor)
print("_______________________________________________19______________________________________________________________")
print(rank_4_tensor.shape)
print(rank_4_tensor.ndim)
print(tf.size(rank_4_tensor))
print("_______________________________________________20______________________________________________________________")

# Get various attributes of our tensor
print("Datatype of every element:", rank_4_tensor.dtype)
print("Number fo dimensions (rank):", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along the 0 axis:", rank_4_tensor.shape[0])
print("Elements along the last axis:", rank_4_tensor.shape[-1])
print("Total number of elements in our tensor:", tf.size(rank_4_tensor))
print("Total number of elements in our tensor:", tf.size(rank_4_tensor).numpy())
print("_______________________________________________21______________________________________________________________")

########################################################################################################################
# Index and expand tensors
########################################################################################################################
# Tensors can be indexed just like python lists

# Get the first 2 elements of each dimension
print(rank_4_tensor[:2, :2, :2, :2])
print("_______________________________________________22______________________________________________________________")

# Get the first element from each dimension from each index except for the final one
print(rank_4_tensor[:1, :1, :1, :])
print("_______________________________________________23______________________________________________________________")

# Create a rank 2 tensor (2 dimensions)
rank_2_tensor = tf.ones(shape=[2, 2])
print(rank_2_tensor)
print(rank_2_tensor.ndim)
print("_______________________________________________24______________________________________________________________")

# Get the last item of each row of our rank 2 tensor
print(rank_2_tensor[:, :-1])
print("_______________________________________________25______________________________________________________________")

# Add in extra dimension to our rank 2 tensor
rank_3_tensor = rank_2_tensor[..., tf.newaxis]
print(rank_3_tensor)
print("_______________________________________________26______________________________________________________________")

# Alternative to tf.newaxis
print(tf.expand_dims(rank_2_tensor, axis=-1))  # "-1" means expand the final axis
print("_______________________________________________27______________________________________________________________")

########################################################################################################################
# Manipulating tensors (tensor operations)
########################################################################################################################
# **basic operations(+, -, *, /)**
"""
you can add values to a tensor using the addition operator
"""

tensor = tf.constant([[10, 7],
                      [3, 4]])
print(tensor + 10)
print("_______________________________________________28______________________________________________________________")

# The original tensor was not changed
print(tensor)
print("_______________________________________________29______________________________________________________________")

# multiplication also works
print(tensor * 10)
print("_______________________________________________30______________________________________________________________")

# subtraction if you want
print(tensor - 10)
print("_______________________________________________31______________________________________________________________")

# we can use the tensorflow built in function too
print(tf.multiply(tensor, 10))
print("_______________________________________________32______________________________________________________________")

########################################################################################################################
# Matrix Multiplication
########################################################################################################################
"""
in machine learning, matrix multiplication is one of them ost common tensor operations

There are 2 rules our tensors (or matrices) need ot fulfil if we're going to matrix multiply them:
1. The inner dimensions must match
2. The resulting matrix has the shape of the outer dimensions
"""
# **resource** info and example of matrix multiplication: https://www.mathsisfun.com/algebra/matrix-multiplying.html

print(tensor)
print("_______________________________________________33______________________________________________________________")
print(tf.matmul(tensor, tensor))
print("_______________________________________________34______________________________________________________________")

# matrix multiplication with python operator "@"
print(tensor @ tensor)
print("_______________________________________________35______________________________________________________________")

# create a tensor (3, 2) tensor.
x = tf.constant([[1, 2],
                 [3, 4],
                 [5, 6]])
# create another (3, 2) tensor.
y = tf.constant([[7, 8],
                 [9, 10],
                 [11, 12]])
print(x, y)
print("_______________________________________________36______________________________________________________________")

# Try to multiply tensors of same shape
# this code will give you a error
# print(x @ y)

# Let's change the shape of y
print(tf.reshape(y, shape=(2, 3)))
print("_______________________________________________37______________________________________________________________")

# Try to matrix multiplication x by reshaped y
reshapey = tf.reshape(y, shape=(2, 3))
print(x @ reshapey)
print("_______________________________________________38______________________________________________________________")

print(tf.matmul(x, reshapey))
print("_______________________________________________39______________________________________________________________")

# Try to change the shape of x instead of y
reshapex = tf.reshape(x, shape=(2, 3))
print(tf.matmul(reshapex, y))
print("_______________________________________________40______________________________________________________________")

# Can do the same with transpose
print(tf.transpose(x))
print("_______________________________________________41______________________________________________________________")

# Try matrix multiplication with transpose rather than reshape
print(tf.matmul(tf.transpose(x), y))

"""
*The dot product*

Matrix multiplication is also referred to as the dot product

you can perform matrix multiplication using:
'tf.matmul()' or
'tf.tensordot()'
"""
# perform the dot product on x and y (requires x or y to be transposed)
print(x, y)
print(tf.tensordot(tf.transpose(x), y, axes=1))
print("_______________________________________________42______________________________________________________________")

# Perform matrix multiplication between x an y (transposed)
print(tf.matmul(x, tf.transpose(y)))
print("_______________________________________________43______________________________________________________________")

# Perform matrix multiplication between x and y (reshaped)
print(tf.matmul(x, tf.reshape(y, shape=(2, 3))))
print("_______________________________________________44______________________________________________________________")

# Check the values of y, reshape y and transposed y
print("normal y: ", y, "\n")
print("reshaped y: ", tf.reshape(y, shape=(2, 3)), "\n")
print("transposed y: ", tf.transpose(y))
print("_______________________________________________45______________________________________________________________")

"""
** Generally, when performing matrix multiplication on two tensors and one of the axis doesn't line up,
    you wil transpose (rather than reshape) one of the tensors to satisfy the matrix multiplication rules. **
"""

########################################################################################################################
# Changing the datatype of a tensor
########################################################################################################################

# Create a new tensor with default datatype (float 32)
b = tf.constant([1.7, 7.4])
print(b.dtype)
print("_______________________________________________46______________________________________________________________")

c = tf.constant([1, 3])
print(c.dtype)
print("_______________________________________________47______________________________________________________________")

# Change from float32 to float16 (reduced precision)
d = tf.cast(b, dtype=tf.float16)
print(d, d.dtype)
print("_______________________________________________48______________________________________________________________")

# Change from int32 to float32
e = tf.cast(c, dtype=tf.float32)
print(e)
print("_______________________________________________49______________________________________________________________")

########################################################################################################################
# Aggregating Tensors
########################################################################################################################
"""
Aggregating tensors = condensing them from multiple values down to a smaller amount of values. 
"""

# Create a new tensor
d = tf.constant([-7, -10])
print(d)
print("_______________________________________________50______________________________________________________________")

# Get the absolute values
print(tf.abs(d))
print("_______________________________________________51______________________________________________________________")

"""
lets go through the following forms of aggregation:
* Get the minimum
* Get the maximum
* Get the mean of a tensor
* Get the sum of a tensor
"""

# Create a random tensor with values between 0 and 100 of size 50
x = tf.constant(np.random.randint(0, 100, size=50))
print(x)
print(tf.size(x), tf.shape(x), x.ndim)
print("_______________________________________________52______________________________________________________________")

# Find the minimum
z = tf.reduce_min(x)
print(z)
print("_______________________________________________53______________________________________________________________")

# Find the maximum
z = tf.reduce_max(x)
print(z)
print("_______________________________________________54______________________________________________________________")

# Find the mean
z = tf.reduce_mean(x)
print(z)
print("_______________________________________________55______________________________________________________________")

# Find the sum
z = tf.reduce_sum(x)
print(z)
print("_______________________________________________56______________________________________________________________")

# **Exercise:** With what we've just learned, find the variance and standard deviation of our x tensor using tensorflow
# Note: to use the variance and std the input tensor needs to be real, so you need to change the datatype to float
z = tf.cast(x, dtype=tf.float16)
print(tf.math.reduce_variance(z))
print(tf.math.reduce_std(z))
print("_______________________________________________57______________________________________________________________")

########################################################################################################################
# Find the positional max and min of a tensor
########################################################################################################################

# create a new tensor for finding positional minimum and maximum
tf.random.set_seed(42)
f = tf.random.uniform(shape=[50])
print(f)
print("_______________________________________________58______________________________________________________________")

# Find the positional maximum
print(tf.argmax(f))
print("_______________________________________________59______________________________________________________________")

# Index on our largest value
print(f[tf.argmax(f)])
print("_______________________________________________60______________________________________________________________")

# Find the max value of F
print(tf.reduce_max(f))
print("_______________________________________________61______________________________________________________________")

# Check for equality
print(f[tf.argmax(f)] == tf.reduce_max(f))
print("_______________________________________________62______________________________________________________________")

# Find the positional minimum
print(tf.argmin(f))
# Find the minimum using the positional minimum index
print(f[tf.argmin(f)])
print("_______________________________________________63______________________________________________________________")

########################################################################################################################
# Squeezing a tensor (removing all single dimensions)
########################################################################################################################

tf.random.set_seed(42)
g = tf.constant(tf.random.uniform(shape=[50]), shape=(1, 1, 1, 1, 50))
print(g)
print("_______________________________________________64______________________________________________________________")

print(g.shape)
g_squeezed = tf.squeeze(g)
print(g_squeezed, g_squeezed.shape)
print("_______________________________________________65______________________________________________________________")

########################################################################################################################
# One-hot encoding tensors
########################################################################################################################

# Create a list of indices
some_list = [0, 1, 2, 3]  # could be red, green, blue, purple

# One-hot encode our list of indices
print(tf.one_hot(some_list, 4))

# Specify custom values for one hot encoding
print(tf.one_hot(some_list, 4, "*", "x"))
print("_______________________________________________66______________________________________________________________")

########################################################################################################################
# trying out more tensor math operations
########################################################################################################################
# Squaring, log, square root

# Create a new tensor
h = tf.range(1, 10)
print(h)
print("_______________________________________________67______________________________________________________________")

# Square a tensor
print(tf.square(h))
print("_______________________________________________68______________________________________________________________")

# Find the square root, note: method requires non int type
print(tf.sqrt(tf.cast(h, dtype=tf.float32)))
print("_______________________________________________69 'nice'_______________________________________________________")

# Find the log, note: method requires non int type
print(tf.math.log(tf.cast(h, dtype=tf.float32)))
print("_______________________________________________70______________________________________________________________")

