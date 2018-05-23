#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Actor critic approach to solving Bipedal walker
# James Kahn 2018

from collections import deque
import os
import random

import keras
# This will need to be conv network
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout, Activation
from keras.optimizers import Adam, SGD

import gym
import numpy as np


# Play some random games at first to get a feel
def play_rand_games():
    env = gym.make('BipedalWalker-v2')
    state_size = env.observation_space.shape
    action_size = env.action_space.shape
    max_steps = 100
    rand_episodes = 10
    rewards = []
    for _ in range(rand_episodes):
        state = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            #env.render()
            action = env.action_space.sample()

            new_state, reward, done, _ = env.step(action)
            #print(new_state, reward)

            episode_reward += reward
            state = new_state

            if done:
                break

        rewards.append(episode_reward)

    print ("Average score over time: " +  str(sum(rewards)/rand_episodes))
    print(rewards)

#play_rand_games()

# Generic controller for deepQ learning
# Sets up network, hyperparams, and 
class Agent():
    def __init__(self, env, state_size, action_size):
        #self.weight_backup      = "cartpole_weight.h5"
        self.weight_backup      = "BipedalWalker_weights.h5"
        self.env                = env
        self.state_size         = state_size
        self.action_size        = action_size
        self.memory             = deque(maxlen=2000)
        self.learning_rate      = 0.001
        #self.decay              = 1e-6
        self.gamma              = 0.95
        # Defines how willing the network is to make a random move
        self.exploration_max    = 1.0
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.995
        self.exploration_rate   = 1.0
        self.brain              = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        #model.add(Dropout(0.25))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        if os.path.isfile(self.weight_backup):
            #model.load_weights(self.weight_backup)
            model = load_model(self.weight_backup)
            self.exploration_rate = self.exploration_min
        return model

    def save_model(self):
            #self.brain.save(self.weight_backup)
            return

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            # Ideally this would return env.action_space.sample()
            # But we don't have the env passed to Agent init
            #return random.randrange(self.action_size)
            return self.env.action_space.sample()
        act_values = self.brain.predict(state)
        #return np.argmax(act_values[0])
        act_values = np.reshape(act_values, (self.action_size))
        return act_values

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        # Only train on small batch of past games to save memory
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            # In case of cartpole this will always be 1 or 0, other games it's different ranges
            target = reward
            # Now predict based on current observation
            # target_f is a [[x, y]] array
            target_f = self.brain.predict(state)
            print('target_f:', target_f)
            print('replay action:', action)

            if not done:
                # First make a prediction on the result of our action (observation after action)
                next_pred = self.brain.predict(next_state)
                print(next_pred[0])

                for i in self.action_size:
                    # Normal deepQ way of doing things (no policy gradients)
                    target = reward + self.gamma * np.amax(next_pred[0][i])
                    target_f[0][i] = target
            self.brain.fit(state, target_f, epochs=1, verbose=0)

        if self.exploration_rate > self.exploration_min:
            # Don't have episode info so can't use epsiode (time)
            #self.exploration_rate = self.exploration_min + (self.exploration_max - self.exploration_min)*np.exp(-self.exploration_decay*episode)
            self.exploration_rate *= self.exploration_decay

class Bipedal_Walker:
    def __init__(self):
        self.sample_batch_size = 32
        self.episodes          = 1000
        self.env               = gym.make('BipedalWalker-v2')

        self.state_size        = self.env.observation_space.shape[0]
        self.action_size       = self.env.action_space.shape[0]
        self.agent             = Agent(self.env, self.state_size, self.action_size)

    def run(self):
        try:
            for index_episode in range(self.episodes):
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])

                done = False
                index = 0
                while not done:
                    #self.env.render()

                    action = self.agent.act(state)
                    #print('run action:', action.shape)

                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.reshape(next_state, [1, self.state_size])
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    index += 1
                print("Episode {}# Score: {}".format(index_episode, index + 1))
                self.agent.replay(self.sample_batch_size)
        finally:
            self.agent.save_model()

if __name__ == "__main__":
    biped = Bipedal_Walker()
    biped.run()
