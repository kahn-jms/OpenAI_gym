#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Make cart go wow
# James Kahn 2018

import gym
import numpy as np
from statistics import mean, median
from collections import Counter

env = gym.make('CartPole-v0')
env.reset()

initial_games = 100
goal_steps = 500
accepted_score = 50

def create_initial_population():
    training_data = []
    accepted_scores = []
    #for _ in range(initial_games):
    # Run until we have enough initial training data
    while len(accepted_scores) < initial_games:
        observation = env.reset()
        prev_observation = [observation]
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

create_initial_population()
