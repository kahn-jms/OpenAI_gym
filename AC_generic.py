#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Generic actor critic training model
# James Kahn 2018

import os
import random
from collections import deque

import gym
import numpy as np

import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam, SGD
from keras import backends as K

import tensorflow as tf


# Model training agent
class AC_Agent():
    def __init__(self, env, sess):
        self.env = env
        self.sess = sess

        self.actor_weight_backup = "actor_weight.h5"
        self.critic_weight_backup = "critic_weight.h5"

        self.memory = deque(maxlen=2000)
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.tau = 0.125

        self.exploration_rate = 1.0
        self.exploration_decay = 0.995

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
        action_input = Input(shape=self.env.observation_space.shape)
        # Why linear activation?
        a1 = Dense(48)(action_input)

        merged = Add([s2, a1])
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
