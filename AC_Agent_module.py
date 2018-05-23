#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Generic actor critic training model
# James Kahn 2018

import os
import random
from collections import deque

import gym
import numpy as np

# import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Activation
from keras.layers.merge import Add
from keras.optimizers import Adam

import tensorflow as tf


# Model training agent
class AC_Agent:
    def __init__(self, env, sess):
        self.env = env
        self.sess = sess

        self.actor_backup = os.path.join('backup', '{}_actor_backup.h5'.format(env.spec.id))
        self.critic_backup = os.path.join('backup', '{}_critic_backup.h5'.format(env.spec.id))

        self.memory = deque(maxlen=2000)
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.tau = 0.125
        self.sample_batch_size = 32

        self.exploration_rate = 1.0
        self.exploration_decay = 0.995

        # Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act
        # Actor model and gradients setup
        self.actor_state_input, self.actor_model = self._build_actor_model()
        _, self.target_actor_model = self._build_actor_model()

        # Where we will feed de/dC (from critic) as input to actor trianing
        self.actor_critic_grad = tf.placeholder(tf.float32, (None, self.env.action_space.shape[0]))

        # Calculate dC/dA (from actor), performs  partial derivatives of actor model outputs
        # w.r.t each of the actor model trainable weights.
        # Output is tensor of length len(trainable_weights)
        # actor_critic_grad sets initial gradients for actor_model.outputs
        self.actor_grads = tf.gradients(
            self.actor_model.outputs,
            self.actor_model.trainable_weights,
            -self.actor_critic_grad)

        grads = zip(self.actor_grads, self.actor_model.trainable_weights)
        # apply_gradients() is second part of minimize() (compute_gradients() is first part)
        # Input; List of (gradient, variable) pairs as returned by compute_gradients()
        # Output: Operation that applies the gradients
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        # Critic model and gradients setup
        self.critic_state_input, self.critic_action_input, self.critic_model = self._build_critic_model()
        _, _, self.target_critic_model = self._build_critic_model()

        # Calculate de/dC (from critic)
        self.critic_grads = tf.gradients(self.critic_model.input, self.critic_action_input)

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
        self.memory.append((state, action, reward, next_state, done))

    def _act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return self.env.action_space.sample()
        act_values = self.actor_model.predict(state)
        return np.argmax(act_values[0])

    def _build_actor_model(self):
        # Need this to return with the model itself
        state_input = Input(shape=self.env.observation_space.shape)
        d1 = Dense(24, activation='relu')(state_input)
        d2 = Dense(48, activation='relu')(d1)
        d3 = Dense(24, activation='relu')(d2)
        # Why shape[0] here?
        output = Dense(self.env.action_space.shape[0], activation='relu')(d3)

        model = Model(input=state_input, output=output)
        adam = Adam(lr=0.001)
        model.compile(loss='mse', optimizer=adam)

        # Recover previous training
        # if os.path.isfile(self.weight_backup):
        #     model.load_weights(self.weight_backup)
        #     # model = load_model(self.weight_backup)
        #     self.exploration_rate = self.exploration_min
        return state_input, model

    def _build_critic_model(self):
        # Need this to return with the model itself
        state_input = Input(shape=self.env.observation_space.shape)
        s1 = Dense(24, activation='relu')(state_input)
        # Why linear activation?
        s2 = Dense(48)(s1)

        # Also need action input to complete dat dere chain rule
        action_input = Input(shape=self.env.action_space.shape)
        # Why linear activation?
        a1 = Dense(48)(action_input)

        merged = Add()([s2, a1])
        m1 = Dense(24, activation='relu')(merged)

        # This is really where the magic is
        # The issue with DQN was that the action space was limited and discrete
        # Threfore we ould just choose the highest reward action and teach the network to
        # chase that
        # Instead we don't train on the decisions of the actor directly, instead we train
        # with the score the critic gives the actor for it's last action?
        output = Dense(1, activation='relu')(m1)

        model = Model(input=[state_input, action_input], output=output)
        adam = Adam(lr=0.001)
        model.compile(loss='mse', optimizer=adam)

        # Recover previous training
        # if os.path.isfile(self.weight_backup):
        #     model.load_weights(self.weight_backup)
        #     # model = load_model(self.weight_backup)
        #     self.exploration_rate = self.exploration_min
        return state_input, action_input, model

    def _train_actor(self, samples):
        for sample in samples:
            state, action, reward, next_state, _ = sample
            # Why doesn't this need an if not done check?
            predicted_action = self.actor_model.predict(state)

            # I don't really understand what's going on here
            grads = self.sess.run(
                self.critic_grads,
                feed_dict={
                    self.critic_state_input: state,
                    self.critic_action_input: predicted_action
                })[0]

            # and here
            self.sess.run(
                self.optimize,
                feed_dict={
                    self.actor_state_input: state,
                    self.actor_critic_grad: grads
                })

    def _train_critic(self, samples):
        for sample in samples:
            state, action, reward, next_state, done = sample
            if not done:
                target_action = self.actor_model.predict(next_state)
                # Predict the reward the next action we would take woud have on the new state
                # resulting from the current action we took.
                # Use this to help the critic look ahead and judge for the highest rewarding step
                future_reward = self.critic_model.predict([next_state, target_action])[0][0]
                reward += self.gamma * future_reward

            reward = np.reshape(reward, (1, 1))
            action = np.reshape(action, (1, self.env.action_space.shape[0]))
            self.critic_model.fit([state, action], reward, verbose=0)

    # Was previously called replay()
    def _train(self):
        if len(self.memory) < self.sample_batch_size:
            return

        # rewards = []
        sample_batch = random.sample(self.memory, self.sample_batch_size)
        self._train_critic(sample_batch)
        self._train_actor(sample_batch)

    def _update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()
        print('amw:', actor_model_weights)
        print('amw:', actor_target_weights)

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.target_actor_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.target_critic_model.set_weights(critic_target_weights)

    def _update_targets(self):
        self._update_actor_target()
        self._update_critic_target()

    # Do everything
    def train_agent(self, episodes, max_steps, render=False, verbose=False):
        try:
            for index_episode in range(episodes):
                state = self.env.reset()

                done = False
                index = 0
                for step in range(max_steps):
                    if render:
                        self.env.render()

                    state = np.reshape(state, (1, self.env.observation_space.shape[0]))

                    action = self._act(state)
                    # action = np.reshape(action, (1, self.env.action_space.shape[0]))

                    next_state, reward, done, _ = self.env.step(action)
                    next_state = next_state.reshape((1, self.env.observation_space.shape[0]))

                    self._remember(state, action, reward, next_state, done)
                    self._train()

                    state = next_state
                    index += 1

                    if done:
                        break

                if verbose:
                    print("Episode {}# Steps: {}".format(index_episode, index))
        finally:
            self._save_models()