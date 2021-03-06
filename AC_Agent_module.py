#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Generic actor critic training model
# James Kahn 2018

import os
import random
from collections import deque

# import gym
import numpy as np

# import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.layers.merge import Add
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

import tensorflow as tf


# Model training agent
class AC_Agent:
    def __init__(self, env, sess):
        self.env = env
        self.sess = sess

        self.actor_backup = os.path.join('backup', '{}_actor_backup.h5'.format(env.spec.id))
        self.critic_backup = os.path.join('backup', '{}_critic_backup.h5'.format(env.spec.id))

        self.memory = deque(maxlen=2000)
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.gamma = 0.99
        self.tau = 0.01
        self.sample_batch_size = 32
        self.validation_batch_size = 16
        self.epochs = 30

        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.exploration_min = 0.10

        # Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act
        # Actor model and gradients setup
        self.actor_state_input, self.actor_model = self._build_actor_model()
        self.actor_params = tf.trainable_variables()

        _, self.target_actor_model = self._build_actor_model()
        self.target_actor_params = tf.trainable_variables()[len(self.actor_params):]

        # Where we will feed de/dC (from critic) as input to actor training
        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.env.action_space.shape[0]])

        # Calculate dC/dA (from actor), performs  partial derivatives of actor model outputs
        # w.r.t each of the actor model trainable weights.
        # Output is tensor of length len(trainable_weights)
        # actor_critic_grad sets initial gradients for actor_model.outputs?
        # I don't fully understand this so it may be what's causing actor to not converge
        # These are unnormalised since we'll process a whole batch
        self.unnormed_actor_grads = tf.gradients(
            self.actor_model.output,
            self.actor_model.trainable_weights,
            -self.actor_critic_grad)
        self.actor_grads = list(map(lambda x: tf.div(x, self.sample_batch_size), self.unnormed_actor_grads))

        grads = zip(self.actor_grads, self.actor_model.trainable_weights)
        # apply_gradients() is second part of minimize() (compute_gradients() is first part)
        # Input; List of (gradient, variable) pairs as returned by compute_gradients()
        # Output: Operation that applies the gradients
        # self.actor_loss = tf.summary.scalar('loss', loss)
        self.optimize = tf.train.AdamOptimizer(self.actor_lr).apply_gradients(grads)

        # Critic model and gradients setup
        self.critic_state_input, self.critic_action_input, self.critic_model = self._build_critic_model()
        self.critic_params = tf.trainable_variables()[(len(self.actor_params) + len(self.target_actor_params)):]

        _, _, self.target_critic_model = self._build_critic_model()
        self.target_critic_params = tf.trainable_variables()[
            (len(self.actor_params) + len(self.target_actor_params) + len(self.critic_params)):]

        # Calculate de/dC (from critic)
        # self.critic_grads = tf.gradients(self.critic_model.input, self.critic_action_input)
        self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input)

        # Initialise everything for gradient calcs
        # I think this should be global_variables_initializer() instead
        # self.sess.run(tf.initialize_all_variables())
        self.sess.run(tf.global_variables_initializer())

    def _save_models(self):
        if not os.path.exists('backup'):
            try:
                os.makedirs('backup')
            except OSError as err:
                print("OS error: {0}".format(err))
        self.actor_model.save(self.actor_backup)
        self.critic_model.save(self.critic_backup)

    def _remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def _act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return self.env.action_space.sample()
        act_values = self.actor_model.predict(state)
        # For discrete choice games:
        # return np.argmax(act_values[0])
        return act_values[0]

    def _build_actor_model(self):
        # Need this to return with the model itself
        state_input = Input(shape=self.env.observation_space.shape)
        d1 = Dense(256, activation='relu')(state_input)
        d2 = Dense(128, activation='relu')(d1)
        # d3 = Dense(64, activation='relu')(d2)
        # Should have different output layer for each action if actions have different ranges,
        # e.g. [-1,1] (tanh) or [0,1] (sigmoid)
        output = Dense(self.env.action_space.shape[0], activation='linear')(d2)

        model = Model(inputs=state_input, outputs=output)
        # adam = Adam(lr=0.001)
        # model.compile(loss='mse', optimizer=adam)

        # Recover previous training
        if os.path.isfile(self.actor_backup):
            model.load_weights(self.actor_backup)
            # model = load_model(self.weight_backup)
            self.exploration_rate = self.exploration_min
        return state_input, model

    def _build_critic_model(self):
        # Need this to return with the model itself
        state_input = Input(shape=self.env.observation_space.shape)
        s1 = Dense(256, activation='relu')(state_input)
        # Why linear activation?
        s2 = Dense(128, activation='relu')(s1)

        # Also need action input to complete dat dere chain rule
        action_input = Input(shape=self.env.action_space.shape)
        # Why linear activation?
        a1 = Dense(128, activation='relu')(action_input)

        merged = Add()([s2, a1])
        m1 = Dense(128, activation='relu')(merged)

        # This is really where the magic is
        # The issue with DQN was that the action space was limited and discrete
        # Threfore we ould just choose the highest reward action and teach the network to
        # chase that
        # Instead we don't train on the decisions of the actor directly, instead we train
        # with the score the critic gives the actor for it's last action?
        # Not sure what activation to put here, need to check output range
        output = Dense(1, activation='linear')(m1)

        model = Model(inputs=[state_input, action_input], outputs=output)
        adam = Adam(lr=self.critic_lr)
        model.compile(loss='mse', optimizer=adam, metrics=['mse', 'mae', 'mape'])

        # Recover previous training
        if os.path.isfile(self.critic_backup):
            model.load_weights(self.critic_backup)
        #     # model = load_model(self.weight_backup)
        #     self.exploration_rate = self.exploration_min
        return state_input, action_input, model

    def _train_actor(self, samples):
        states = np.asarray([s[0] for s in samples])
        actions = np.zeros((len(samples), self.env.action_space.shape[0]))
        # actions = np.asarray([s[1] for s in samples])
        for idx, sample in enumerate(samples):
            state, action, reward, next_state, _ = sample
        #     # Why not just use the actual action used?
            predicted_action = self.actor_model.predict(state)
        #     # predicted_action = action
        #     # predicted_action = np.reshape(predicted_action, (1, self.env.action_space.shape[0]))
            actions[idx] = predicted_action

        states = np.reshape(states, (len(samples), self.env.observation_space.shape[0]))
        # actions = np.reshape(actions, (len(samples), self.env.action_space.shape[0]))
        # I don't really understand what's going on here
        grads = self.sess.run(
            self.critic_grads,
            feed_dict={
                self.critic_state_input: states,
                # self.critic_action_input: predicted_action
                self.critic_action_input: actions
            })[0]

        # This is basically an actor_model.fit?
        self.sess.run(
            self.optimize,
            feed_dict={
                self.actor_state_input: states,
                self.actor_critic_grad: grads
            }
        )

    def _train_critic(self, samples):
        states = np.asarray([s[0] for s in samples])
        actions = np.asarray([s[1] for s in samples])
        rewards = np.zeros((len(samples), 1))
        for idx, sample in enumerate(samples):
            state, action, reward, next_state, done = sample
            if not done:
                target_action = self.target_actor_model.predict(next_state)
                # Predict the reward the next action we would take woud have on the new state
                # resulting from the current action we took.
                # Use this to help the critic look ahead and judge for the highest rewarding step
                target_future_reward = self.target_critic_model.predict([next_state, target_action])[0][0]
                reward += self.gamma * target_future_reward

            rewards[idx] = reward

        states = np.reshape(states, (len(samples), self.env.observation_space.shape[0]))
        actions = np.reshape(actions, (len(samples), self.env.action_space.shape[0]))
        # rewards = np.reshape(reward, (None, 1))

        # Adding tensorboard callback for review
        # tbCallBack = TensorBoard(
        #     log_dir='logs',
        #     histogram_freq=10,
        #     write_graph=False,
        #     write_grads=True,
        #     batch_size=self.validation_batch_size,
        #     write_images=False,
        # )

        # self.critic_model.fit(
        #     [states[:self.sample_batch_size], actions[:self.sample_batch_size]],
        #     rewards[:self.sample_batch_size],
        #     batch_size=self.sample_batch_size,
        #     verbose=0,
        #     # validation_data=(
        #     #     [states[self.sample_batch_size:], actions[self.sample_batch_size:]],
        #     #     rewards[self.sample_batch_size:]
        #     # ),
        #     # callbacks=[tbCallBack],
        # )
        critic_metrics = self.critic_model.train_on_batch(
            [states[:self.sample_batch_size], actions[:self.sample_batch_size]],
            rewards[:self.sample_batch_size],
        )
        # print('Critic: ', end='')
        # print(self.critic_model.metrics_names[0], ': ', critic_metrics[0], ' | ', end='')
        # print(self.critic_model.metrics_names[1], ': ', critic_metrics[1], ' | ', end='')
        # print(self.critic_model.metrics_names[2], ': ', critic_metrics[2], ' | ', end='')
        # print(self.critic_model.metrics_names[3], ': ', critic_metrics[3], ' | ')
        # for i, name in enumerate(self.critic_model.metrics_names):
        #     print(name, ': ', critic_metrics[i], ' | ', end='')
        return critic_metrics

    # Was previously called replay()
    def _train(self):
        if len(self.memory) < (self.sample_batch_size + self.validation_batch_size):
            return

        # rewards = []
        sample_batch = random.sample(self.memory, (self.sample_batch_size + self.validation_batch_size))
        self._train_critic(sample_batch)
        self._train_actor(sample_batch)

        # Putting exploration decay here for now, could also only decay after each episode
        # Really depends on how quickly an episode goes
        # Eventually want to use exponential decay instead
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

    def _update_actor_target(self):
        # actor_model_weights = self.actor_model.get_weights()
        # actor_target_weights = self.target_actor_model.get_weights()
        # # print('amw:', actor_model_weights)
        # # print('amw:', actor_target_weights)
        # for i in range(len(actor_target_weights)):
        #     # Need the target network to follow main network with some delay otherwise training unstable
        #     actor_target_weights[i] = (self.tau * actor_model_weights[i] +
        #                                (1 - self.tau) * actor_target_weights[i])
        # self.target_actor_model.set_weights(actor_target_weights)
        [self.target_actor_params[i].assign(tf.multiply(self.actor_params[i], self.tau) +
                                            tf.multiply(self.target_actor_params[i], 1. - self.tau))
         for i in range(len(self.target_actor_params))]

    def _update_critic_target(self):
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()
        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = (self.tau * critic_model_weights[i] +
                                        (1. - self.tau) * critic_target_weights[i])
        self.target_critic_model.set_weights(critic_target_weights)

    def _update_targets(self):
        self._update_actor_target()
        self._update_critic_target()

    # Do everything
    def train_agent(self, episodes, max_steps, render=False, render_freq=100, verbose=False):
        try:
            for index_episode in range(episodes):
                state = self.env.reset()

                done = False
                index = 0
                tot_reward = 0
                for step in range(max_steps):
                    if render and (index_episode % render_freq == 0):
                        if step % 5 == 0:
                            self.env.render()

                    state = np.reshape(state, (1, self.env.observation_space.shape[0]))

                    action = self._act(state)
                    # action = np.reshape(action, (1, self.env.action_space.shape[0]))

                    next_state, reward, done, _ = self.env.step(action)
                    next_state = next_state.reshape((1, self.env.observation_space.shape[0]))
                    self._remember(state, action, reward, next_state, done)

                    self._train()
                    self._update_targets()

                    state = next_state
                    index += 1
                    tot_reward += reward

                    if done:
                        break

                if verbose:
                    print("Episode {}# Steps: {} Reward: {}".format(index_episode, index, tot_reward))
                    # print('exploration rate: {}'.format(self.exploration_rate))

                # Try training after each episode with bigger batch size
                # for _ in range(self.epochs):
                #     self._train()
                #     self._update_targets()
        finally:
            self._save_models()
