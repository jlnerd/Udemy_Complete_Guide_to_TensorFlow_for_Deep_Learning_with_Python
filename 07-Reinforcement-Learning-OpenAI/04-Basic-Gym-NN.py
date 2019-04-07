import tensorflow as tf
import gym
import numpy as np

n_inputs = 4
n_HLs =4
n_outputs = 1

initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape  = [None, n_inputs])

HL1 = tf.layers.dense(X,n_HLs, activation=tf.nn.relu, kernel_initializer=initializer)
HL2 = tf.layers.dense(HL1,n_HLs, activation=tf.nn.relu, kernel_initializer=initializer)

output = tf.layers.dense(HL2, n_outputs, activation = tf.nn.sigmoid, kernel_initializer = initializer)

probabilties = tf.concat(axis=1, values=[output, 1-output])

action = tf.multinomial(probabilties, num_samples=1)

init = tf.global_variables_initializer()

episodes = 50
steps = 500

env = gym.make('CartPole-v0')

avg_steps = []
with tf.Session() as sess:

    sess.run(init)

    for i_episode in range(episodes):
        obs = env.reset()

        for step in range(steps):
            action_val = action.eval(feed_dict = {X:obs.reshape(1,n_inputs)})
            obs, reward, done, info = env.step(action_val[0][0]) #returns 0 or 1

            if done:
                avg_steps.append(step)
                print("Done After {} Steps".format(step))
                break

print("After {} episodes, average steps per game was {} ".format(episodes,np.mean(avg_steps)))

env.close()
