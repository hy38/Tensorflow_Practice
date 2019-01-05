import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.Variable(5.0)
# W = tf.Variable(-3.0)

# Our hypothesis for linear model X * W
hypothesis = X * W

# Manual gradient
gradient = tf.reduce_mean((W * X - Y) * X) * 2

# cost / Loss function
cost = tf.reduce_mean(tf.square(hypothesis-Y))   # sum vs mean

# Minimize: Gradient Descent using derivative: W -= Learning_rate * derivate
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# Get gradients
gvs = optimizer.compute_gradients(cost)

# Apply gradients
apply_gradients = optimizer.apply_gradients(gvs)

# Launch the graph in a session.
sess = tf.Session()

# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)
