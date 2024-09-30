############### BSDE NN - to be tested code #######################################################

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from time import time

# Set data type
DTYPE='float32'
#DTYPE='float64'
tf.keras.backend.set_floatx(DTYPE)
print('TensorFlow version used: {}'.format(tf.__version__))

# Final time
T = 1.

# Spatial dimensions
dim = 100

# Number of equidistant intervals in time
N = 20

# Derive time step size and t_space
dt = T/N
t_space = np.linspace(0, T, N + 1)

# Point-of-interest at t=0
x = np.zeros(dim)

# Diffusive term is assumed to be constant
sigma = np.sqrt(2)

def draw_X_and_dW(num_sample, x):
    """ Function to draw num_sample many paths of the stochastic process X
    and the corresponding increments of simulated Brownian motions dW. """

    dim = x.shape[0]

    # Draw all increments of W at once
    dW = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=(num_sample, dim, N)).astype(DTYPE)

    # Initialize the array X
    X = np.zeros((num_sample, dim, N+1), dtype=DTYPE)

    # Set starting point to x for each draw
    X[:, :, 0] = np.ones((num_sample, dim)) * x

    for i in range(N):
        # This corresponds to the Euler-Maruyama Scheme
        X[:, :, i+1] = X[:, :, i] + sigma * dW[:, :, i]

    # Return simulated paths as well as increments of Brownian motion
    return X, dW


num_sample=10

# Draw 10 sample paths
X,dW = draw_X_and_dW(num_sample, np.zeros(1))


# Plot these paths
fig,ax = plt.subplots(1)
for i in range(num_sample):
    ax.plot(t_space,X[i,0,:])
ax.set_xlabel('$t$')
ax.set_ylabel('$X_t$');

#NN code
class BSDEModel(tf.keras.Model):
    def __init__(self, **kwargs):

        # Call initializer of tf.keras.Model
        super().__init__(**kwargs)

        # Initialize the value u(0, x) randomly
        u0 = np.random.uniform(.1, .3, size=(1)).astype(DTYPE)
        self.u0 = tf.Variable(u0)

        # Initialize the gradient nabla u(0, x) randomly
        gradu0 = np.random.uniform(-1e-1, 1e-1, size=(1, dim)).astype(DTYPE)
        self.gradu0 = tf.Variable(gradu0)

        # Create template of dense layer without bias and activation
        _dense = lambda dim: tf.keras.layers.Dense(
            units=dim,
            activation=None,
            use_bias=False)

        # Create template of batch normalization layer
        _bn = lambda : tf.keras.layers.BatchNormalization(
            momentum=.99,
            epsilon=1e-6,
            beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
            gamma_initializer=tf.random_uniform_initializer(0.1, 0.5))


        # Initialize a list of networks approximating the gradient of u(t, x) at t_i
        self.gradui = []

        # Loop over number of time steps
        for _ in range(N - 1):

            # Batch normalization on dim-dimensional input
            this_grad = tf.keras.Sequential()
            this_grad.add(tf.keras.layers.Input(shape=(dim,)))
            this_grad.add(_bn())

            # Two hidden layers of type (Dense -> Batch Normalization -> ReLU)
            for _ in range(2):
                this_grad.add(_dense(dim+10))
                this_grad.add(_bn())
                this_grad.add(tf.keras.layers.ReLU())

            # Dense layer followed by batch normalization for output
            this_grad.add(_dense(dim))
            this_grad.add(_bn())
            self.gradui.append(this_grad)


def simulate_Y(inp, model, fun_f):
    """ This function performs the forward sweep through the network.
    Inputs:
        inp - (X, dW)
        model - model of neural network, contains
            - u0  - variable approximating u(0, x)
            - gradu0 - variable approximating nabla u(0, x)
            - gradui - list of NNs approximating the mapping: x -> nabla u(t_i, x)
        fun_f - function handle for cost function f
    """

    X, dW = inp
    num_sample = X.shape[0]


    e_num_sample = tf.ones(shape=[num_sample, 1], dtype=DTYPE)

    # Value approximation at t0
    y = e_num_sample * model.u0

    # Gradient approximation at t0
    z = e_num_sample * model.gradu0

    for i in range(N-1):
        t = t_space[i]

        # Determine terms in right-hand side of Y-update at t_i
        eta1 = - fun_f(t, X[:, :, i], y, z) * dt
        eta2 = tf.reduce_sum(z * dW[:, :, i], axis=1, keepdims=True)

        # Compute new value approximations at t_{i+1}
        y = y + eta1 + eta2

        # Obtain gradient approximations at t_{i+1}
        # Scaling the variable z by 1/dim improves the convergence properties
        # and has been used in the original code https://github.com/frankhan91/DeepBSDE
        # z still approximates \sigma^T \nabla u, but the network learns to represent
        # a scaled version.
        z = model.gradui[i](X[:, :, i + 1]) / dim


    # Final step
    eta1 = - fun_f(t_space[N-1], X[:, :, N-1], y, z) * dt
    eta2 = tf.reduce_sum(z * dW[:, :, N-1], axis=1, keepdims=True)
    y = y + eta1 + eta2

    return y

#loss fn to be min

def loss_fn(inp, model, fun_f, fun_g):
    """ This function computes the mean-squarred error of the difference of Y_T and g(X_T)
    Inputs:
        inp - (X, dW)
        model - model of neural network containing u0, gradu0, gradui
        fun_f - function handle for cost function f
        fun_g - function handle for terminal condition g
    """
    X, _ = inp

    # Forward pass to compute value estimates
    y_pred = simulate_Y(inp, model, fun_f)

    # Final time condition, i.e., evaluate g(X_T)
    y = fun_g(X[:, :, -1])

    # Compute mean squared error
    y_diff = y-y_pred
    loss = tf.reduce_mean(tf.square(y_diff))

    return loss

#gradient o floss fn
@tf.function
def compute_grad(inp, model, fun_f, fun_g):
    """ This function computes the gradient of the loss function w.r.t.
    the trainable variables theta.
    Inputs:
        inp - (X, dW)
        model - model of neural network containing u0, gradu0, gradui
        fun_f - function handle for cost function f
        fun_g - function handle for terminal condition g
    """

    with tf.GradientTape() as tape:
        loss = loss_fn(inp, model, fun_f, fun_g)

    grad = tape.gradient(loss, model.trainable_variables)

    return loss, grad

#define fn f and g (0 drift in x, dt = 2)
# Define cost function f, remember that z approximates \sigma^T \nabla u
def fun_f(t, x, y, z):
    return - tf.reduce_sum(tf.square(z), axis=1, keepdims=True) / (sigma**2)

# Set terminal value function g
def fun_g(x):
    return tf.math.log( (1+tf.reduce_sum(tf.square(x), axis=1, keepdims=True)) / 2)

# Set learning rate
lr = 1e-2
# Choose optimizer for gradient descent step
optimizer = tf.keras.optimizers.Adam(lr, epsilon=1e-8)

# Initialize neural network architecture
model = BSDEModel()
y_star = 4.59016

# Initialize list containing history of losses
history = []

#train epoch
t0 = time()

num_epochs = 40000

# Initialize header of output
print('  Iter        Loss        y   L1_rel    L1_abs   |   Time  Stepsize')

for i in range(num_epochs):

    # Each epoch we draw a batch of 64 random paths
    X, dW = draw_X_and_dW(64, x)

    # Compute the loss as well as the gradient
    loss, grad = compute_grad((X, dW), model, fun_f, fun_g)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))

    # Get current Y_0 \approx u(0,x)
    y = model.u0.numpy()[0]

    currtime = time() - t0
    l1abs = np.abs(y - y_star)
    l1rel = l1abs / y_star

    hentry = (i, loss.numpy(), y, l1rel, l1abs, currtime, lr)
    history.append(hentry)
    if i%10 == 0:
        print('{:5d} {:12.4f} {:8.4f} {:8.4f}  {:8.4f}   | {:6.1f}  {:6.2e}'.format(*hentry))

#plot results
fig, ax = plt.subplots(1,2,figsize=(15,6))
xrange = range(len(history))
ax[0].semilogy(xrange, [e[1] for e in history],'k-')
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('training loss')
ax[1].plot(xrange, [e[2] for e in history])
ax[1].set_xlabel('epoch')
ax[1].set_ylabel('$u(0,x)$');
