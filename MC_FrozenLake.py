#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# MC approach to training for episodic games with fixed game space (e.g. chess board)
# James Kahn 2018

from collections import deque
import os

import gym
import numpy as np

#env = gym.make('FrozenLake-v0')
env = gym.make('FrozenLake8x8-v0')

state_size = env.observation_space.n
action_size = env.action_space.n


lr = 1.0
gamma              = 0.95
exploration_max    = 1.0
exploration_min    = 0.01
exploration_decay  = 0.05
exploration_rate   = exploration_max

max_steps = 150
train_episodes = 20000

def act(state):
    if np.random.rand() <= exploration_rate:
        # Ideally this would return env.action_space.sample()
        # But we don't have the env passed to Agent init
        return env.action_space.sample()
    else:
        return np.argmax(qtable[state:])

# Ideally would add some influence to win in minimum number of steps
def aggressive_reward(reward, done):
    if not done and reward == 0:
        return -1
    elif reward == 1:
        return 10
    else:
        return -10

#def remember(self, state, action, reward, next_state, done):
#    self.memory.append((state, action, reward, next_state, done))

def train_Qtable():
    qtable_backup = "MC_frozenLake_qtable.npy"
    qtable = np.zeros((state_size, action_size))
    rewards = []
    global lr

    for episode in range(train_episodes):
        state = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            action = act(state)

            new_state, reward, done, _ = env.step(action)

            #mod_reward = aggressive_reward(reward, done)

            qtable[state, action] = qtable[state, action] + \
                    lr * (reward + gamma*np.max(qtable[new_state,:]) \
                          - qtable[state, action])
            #  Try a more aggressive reward scheme
            #qtable[state, action] = qtable[state, action] + \
            #        lr * (mod_reward + gamma*np.max(qtable[new_state,:]) \
            #              - qtable[state, action])

            #episode_reward += mod_reward
            episode_reward += reward
            state = new_state

            if done:
                break

        rewards.append(episode_reward)

        # Should really use an exponential decay rate I guess
        #if exploration_rate > exploration_min:
        #    exploration_rate *= exploration_decay
        exploration_rate = exploration_min + (exploration_max - exploration_min)*np.exp(-exploration_decay*episode)

        # Reduce learning rate  over time
        if episode%5000 == 0:
            lr *= 0.5



    print ("Score over time: " +  str(sum(rewards)/train_episodes))
    print(qtable)
    return qtable

def play_game(qtable, ngames = 10):
    scores = []
    wins = []

    # Play the games
    for episode in range(ngames):
        state = env.reset()
        score = 0
        win = 0

        for _ in range(max_steps):
            #env.render()

            action = np.argmax(qtable[state,:])
            state, reward, done, _ = env.step(action)

            #mod_reward = aggressive_reward(reward, done)
            #score += mod_reward
            win += reward

            if done:
                break

        scores.append(score)
        wins.append(win)
        #print("Episode {}# Score: {}".format(episode, score))
    #return scores, wins
    return wins


if __name__ == "__main__":
    qtable = train_Qtable()
    wins = play_game(qtable, 100)

    #print('Average Score over {} games:'.format(len(scores)), sum(scores)/len(scores))
    #print(scores)
    print('Average wins over {} games:'.format(len(wins)), sum(wins)/len(wins))
    print(wins)
