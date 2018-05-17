#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Make cart go wow
# James Kahn 2018

import os
import time
import argparse

import gym
import numpy as np
from statistics import mean, median
from collections import Counter

import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout, Activation
from keras.optimizers import SGD
#from keras import backend as K

env = gym.make('CartPole-v0')
env.reset()

initial_games = 300
goal_steps = 500
accepted_score = 50

def read_args():
    parser = argparse.ArgumentParser(
        description='''CartPole game player''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-m', type=str, required=False,
                        help="Path to existing trained model.", metavar="MODEL",
                        dest='model')
    return parser.parse_args()

def create_initial_population():
    training_data = []
    accepted_scores = []
    #for _ in range(initial_games):
    # Run until we have enough initial training data
    while len(accepted_scores) < initial_games:
        observation = env.reset()
        prev_observation = observation
        score = 0
        game_memory = []
        for _ in range(goal_steps):
            #env.render()
            #print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            # Save action taken given previous status
            game_memory.append([prev_observation, action])
            prev_observation = observation
            score += 1

            if done:
                #print("Episode finished after {} timesteps".format(score))
                break
        # Only accept games that are good enough
        if score >= accepted_score:
            accepted_scores.append(score)

            # Convert training targets to one-hot and save to training_data
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                else:
                    output = [1,0]
                training_data.append([data[0], output])

    training_data_save = np.array(training_data)
    np.save('initial_population.npy', training_data_save)

    print('Average accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data


def keras_fully_connected(input_size):

    model = Sequential()
    model.add(Dense(24, activation='relu', input_dim=input_size))
    model.add(Dropout(0.5))
    model.add(Dense(24, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    return model

def train_model(training_data, model=False):

    training_frac = 0.9

    x_train = np.array([i[0] for i in training_data])
    # Dunno why we need this let's see
    #x_train = x_train.reshape(-1, len(training_data[0][0]), 1)

    # Shouldn't this also be a numpy array?
    y_train = np.array([i[1] for i in training_data])

    # Split into training and testing data
    x_test = x_train[int(len(x_train)*training_frac):]
    y_test = y_train[int(len(y_train)*training_frac):]
    x_train = x_train[:int(len(x_train)*training_frac)]
    y_train = y_train[:int(len(y_train)*training_frac)]

    if not model:
        model = keras_fully_connected(input_size = len(x_train[0]))

    now = time.strftime("%Y.%m.%d.%H.%M")
    tbCallBack = keras.callbacks.TensorBoard(
        log_dir = os.path.join('logs',now),
        histogram_freq = 1,
        write_graph = True,
        write_grads = True,
        #batch_size = batch_size,
        batch_size = 32,
        write_images = True,
    )

    model.fit(
        x_train,
        y_train,
        epochs=20,
        batch_size=128,
        #validation_data=(x_test,y_test),
        #callbacks=[tbCallBack],
    )
    score = model.evaluate(x_test, y_test, batch_size=128)
    print('Model score:', score)

    # Save the model to speed up processing later
    model.save('trained_nn.h5')

    return model


def play_nn_games(model):
    scores = []
    choices = []

    # Play the games
    for _ in range(10):
        observation = env.reset()
        prev_observation = observation
        score = 0
        game_memory = []

        for _ in range(goal_steps):
            #env.render()

            prev_obs_np = np.array([prev_observation])#.reshape(-1, len(prev_observation))
            action = model.predict(prev_obs_np)
            # Just one a 1 or 0 as output
            action = np.argmax(action[0])

            choices.append(action)

            observation, reward, done, info = env.step(action)
            prev_observation = observation
            game_memory.append([observation, action])

            score += reward
            if done:
                break

        scores.append(score)
    return scores, choices

args = read_args()
if not args.model:
    train_data = create_initial_population()
    model = train_model(train_data)
else:
    model = load_model(args.model)

scores, choices = play_nn_games(model)

print('Average Score:',sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
