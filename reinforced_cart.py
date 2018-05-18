#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copy of self teaching network from 
# https://medium.com/@gtnjuvin/my-journey-into-deep-q-learning-with-keras-and-gym-3e779cc12762
# This version looks two steps into the past to make prediction of next step.
# Ideally this would be enough steps to handle momentum but this is hard to handle in the beginning I guess
# In the simple case of CartPole this could just be random moves?
# James Kahn 2018

from collections import deque
import os
import random

import gym
import numpy as np

import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout, Activation
from keras.optimizers import Adam, SGD

class Agent():
    def __init__(self, state_size, action_size):
        #self.weight_backup      = "cartpole_weight.h5"
        self.weight_backup      = "reinforced_cartpole.h5"
        self.state_size         = state_size
        self.action_size        = action_size
        self.memory             = deque(maxlen=2000)
        self.learning_rate      = 0.001
        #self.decay              = 1e-6
        self.gamma              = 0.95
        # Defines how willing the network is to make a random move
        self.exploration_rate   = 1.0
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.995
        self.brain              = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(32, input_dim=2*self.state_size, activation='relu'))
        #model.add(Dropout(0.25))
        model.add(Dense(32, activation='relu'))
        #model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        #sgd = SGD(lr=self.learning_rate,
        #          decay=self.decay,
        #          momentum=0.9,
        #          nesterov=True)
        #model.compile(loss='categorical_crossentropy',
        #              optimizer=sgd,
        #              metrics=['accuracy'])

        # If weve already done some training, load prev weights
        if os.path.isfile(self.weight_backup):
            #model.load_weights(self.weight_backup)
            model = load_model(self.weight_backup)
            self.exploration_rate = self.exploration_min
        return model

    def save_model(self):
            self.brain.save(self.weight_backup)

    def act(self, state, prev_state):
        if np.random.rand() <= self.exploration_rate:
            # Ideally this would return env.action_space.sample()
            # But we don't have the env passed to Agent init
            return random.randrange(self.action_size)
        act_values = self.brain.predict(np.append(prev_state, state, axis=1))
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, prev_state, next_state, done):
        self.memory.append((state, action, reward, prev_state, next_state, done))

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        # Only train on small batch of past games to save memory
        # Should be able to call states directly from self.memory, just need to make sure
        # i don't grabe the first or last play of a game, otherwise will get state from a
        # different game
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, prev_state, next_state, done in sample_batch:
            # In case of cartpole this will always be 1 or 0, other games it's different ranges
            target = reward
            if not done:
                # First make a prediction on the result of our action (observation after action)
                # This tells us if we will need to move left or right after the next move?
                next_pred = self.brain.predict(np.append(state, next_state, axis=1))
                #print(next_pred)
                # Add this (weighted) to the reward (1) 
                # Why is this necessary? Why not just gamma * pred? that is ~21 anyway
                target = reward + self.gamma * np.amax(next_pred[0])
            # Now predict based on current observation
            # target_f is a [[x, y]] array
            target_f = self.brain.predict(np.append(prev_state, state, axis=1))
            # Whichever element in the one-hot output was the actual resulting action from the observation (0 or 1), replace that
            # element with the weighted prediction from the following step, this is supposed to encourage survival
            # If the next step doesn't exist, it means this action killed us, so only replaced with 1, a.k.a heavily
            # bias the training to do the opposite.
            # This is all a way to see into the result of future steps, as training continues this let's the network
            # slowly strengthen successful pathways and weaken the failing ones.
            # Can it be extended to include previous steps too? Or two steps into the future?
            target_f[0][action] = target
            self.brain.fit(np.append(prev_state, state, axis=1), target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

class CartPole:
    def __init__(self):
        self.sample_batch_size = 128
        self.episodes          = 1000
        self.env               = gym.make('CartPole-v1')

        self.state_size        = self.env.observation_space.shape[0]
        self.action_size       = self.env.action_space.n
        self.agent             = Agent(self.state_size, self.action_size)


    def run(self):
        try:
            for index_episode in range(self.episodes):
                prev_state = np.zeros((1, self.state_size))
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])

                done = False
                index = 0
                while not done:
#                    self.env.render()
                    action = self.agent.act(state, prev_state)

                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.reshape(next_state, [1, self.state_size])
                    self.agent.remember(state, action, reward, prev_state, next_state, done)
                    prev_state = state
                    state = next_state
                    index += 1
                print("Episode {}# Score: {}".format(index_episode, index + 1))
                self.agent.replay(self.sample_batch_size)
        finally:
            self.agent.save_model()

if __name__ == "__main__":
    cartpole = CartPole()
    cartpole.run()
