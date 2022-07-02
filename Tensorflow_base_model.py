## pip install tensorflow


Try modeling a non-linear function such as:  𝑦=𝑥𝑒−𝑥2


# Here we'll import data processing libraries like Numpy and Tensorflow
import numpy as np
import tensorflow as tf
# Use matplotlib for visualizing the model
from matplotlib import pyplot as plt

# Here we'll show the currently installed version of TensorFlow
print(tf.__version__)



## Input variable 
X = tf.constant(np.linspace(0, 2, 1000), dtype=tf.float32)

## Actual Output
Y = X * tf.exp(-X**2)


## Plot the function here now 
%matplotlib inline

# The .plot() is a versatile function, and will take an arbitrary number of arguments. For example, to plot x versus y.
plt.plot(X, Y)





# Let's make_features() procedure | And stack the features into output 
def make_features(X):
    # The tf.ones_like() method will create a tensor of all ones that has the same shape as the input.
    f1 = tf.ones_like(X)
    f2 = X
    
    # The tf.square() method will compute square of input tensor element-wise.
    f3 = tf.square(X)
    
    # The tf.sqrt() method will compute element-wise square root of the input tensor.
    f4 = tf.sqrt(X)
    
    # The tf.exp() method will compute exponential of input tensor element-wise.
    f5 = tf.exp(X)
    
    # The tf.stack() method will stacks a list of rank-R tensors into one rank-(R+1) tensor.
    return tf.stack([f1, f2, f3, f4, f5], axis=1)



# Let's define predict() procedure that will remove dimensions of size 1 from the shape of a tensor.
def predict(X, W):
    return tf.squeeze(X @ W, -1)



## Lets define the loss here now 
def loss_mse(X,Y,W):
	Y_hat = predict(X,W)
	errors = (Y_hat - Y)**2
    return tf.reduce_mean(errors)


## define compute grandient 
def compute_gradients(X,Y,W):
	with tf.GradientTape() as tape:
		 loss = loss_mse(Xf, Y, W)
    return tape.gradient(loss, W)


############################################################################
############################################################################
## starts the training here now !! 
############################################################################
############################################################################

STEPS = 2000
LEARNING_RATE = .02


Xf = make_features(X)
n_weights = Xf.shape[1]

W = tf.Variable(np.zeros((n_weights, 1)), dtype=tf.float32)


# For plotting
steps, losses = [], []
plt.figure()


for step in range(1, STEPS + 1):

    ## gradient weight here now 
    dW = compute_gradients(X, Y, W)
    W.assign_sub(dW * LEARNING_RATE)

    if step % 100 == 0:
        loss = loss_mse(Xf, Y, W)
        steps.append(step)
        losses.append(loss)
        plt.clf()
        plt.plot(steps, losses)


print("STEP: {} MSE: {}".format(STEPS, loss_mse(Xf, Y, W)))


############################################################################
############################################################################
## training completed here now !!! 
############################################################################
############################################################################


# The .figure() method will create a new figure, or activate an existing figure.
plt.figure()
# The .plot() is a versatile function, and will take an arbitrary number of arguments. For example, to plot x versus y.
plt.plot(X, Y, label='actual')
plt.plot(X, predict(Xf, W), label='predicted')
# The .legend() method will place a legend on the axes.
plt.legend()




