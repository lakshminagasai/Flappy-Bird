#!/usr/bin/env python
from __future__ import print_function

import os
import tensorflow as tf


import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
import time

# Disable oneDNN optimizations warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Disable eager execution for TensorFlow 2.x compatibility with TensorFlow 1.x code
tf.compat.v1.disable_eager_execution()

GAME = 'bird'
ACTIONS = 2  # number of valid actions (e.g., jump, do nothing)
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 100000.  # timesteps to observe before training
EXPLORE = 2000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.01  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 32  # size of minibatch
FRAME_PER_ACTION = 1

MAX_TIMESTEPS = 1000  # Maximum number of timesteps
MAX_TIME = 60  # Time limit in seconds (e.g., 1 hour)

def epsilon_greedy_policy(epsilon, state, readout, s):
    if np.random.rand() <= epsilon:
        return random.randint(0, ACTIONS - 1)  # Random action
    else:
        return np.argmax(readout.eval(feed_dict={s: [state]}))  # Best action according to Q-values


# Define the weight_variable and bias_variable functions
def weight_variable(shape):
    initial = tf.compat.v1.truncated_normal(shape, stddev=0.01)
    return tf.compat.v1.Variable(initial)

def bias_variable(shape):
    initial = tf.compat.v1.constant(0.01, shape=shape)
    return tf.compat.v1.Variable(initial)

# Define the convolution and max pooling layers
def conv2d(x, W, stride):
    return tf.compat.v1.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

def max_pool_2x2(x):
    return tf.compat.v1.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# Define the network architecture
def createNetwork():
    # Define the network weights and layers
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # Input layer
    s = tf.compat.v1.placeholder("float", [None, 80, 80, 4])

    # Hidden layers
    h_conv1 = tf.compat.v1.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.compat.v1.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)

    h_conv3 = tf.compat.v1.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

    h_conv3_flat = tf.compat.v1.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # Readout layer
    readout = tf.compat.v1.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

# Define the training process
def trainNetwork(sess):
    s, readout, h_fc1 = createNetwork()  # Create the network

    a = tf.compat.v1.placeholder("float", [None, ACTIONS])
    y = tf.compat.v1.placeholder("float", [None])
    readout_action = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(readout, a), reduction_indices=1)
    cost = tf.compat.v1.reduce_mean(tf.compat.v1.square(y - readout_action))
    train_step = tf.compat.v1.train.AdamOptimizer(1e-6).minimize(cost)

    # Initialize game state
    game_state = game.GameState()

    # Initialize replay memory
    D = deque()

    # Initialize TensorFlow session and network
    saver = tf.compat.v1.train.Saver()
    sess.run(tf.compat.v1.global_variables_initializer())
    checkpoint = tf.compat.v1.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    epsilon = INITIAL_EPSILON
    t = 0
    start_time = time.time()

    # Initialize the first state (s_t)
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)  # Stack the state to match 4 frames

    while t < MAX_TIMESTEPS:
        if time.time() - start_time >= MAX_TIME:
            print("Time limit reached. Stopping training.")
            break

        # Use the random policy
        action_index = epsilon_greedy_policy(epsilon, s_t, readout, s)

        a_t = np.zeros([ACTIONS])
        a_t[action_index] = 1

        # Get the next state from the game
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)  # Stack the new state with the previous ones

        D.append((s_t, a_t, r_t, s_t1, terminal))  # Store the experience in replay memory
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        if t > OBSERVE:
            # Sample a minibatch for training
            minibatch = random.sample(D, BATCH)
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict={s: s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            train_step.run(feed_dict={y: y_batch, a: a_batch, s: s_j_batch})

        s_t = s_t1  # Update the state
        t += 1

        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step=t)

        state = "explore" if t <= OBSERVE else "train"
        # Print policy and some additional information (like epsilon and Q values)
        print(f"TIMESTEP {t} / STATE {state} / EPSILON {epsilon} / ACTION {action_index} / REWARD {r_t} / Q_MAX {np.max(readout.eval(feed_dict={s: [s_t]}))}")

def playGame():
    sess = tf.compat.v1.InteractiveSession()
    trainNetwork(sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
