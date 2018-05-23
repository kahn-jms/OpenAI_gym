#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Actor critic approach to solving Bipedal walker
# James Kahn 2018

import gym

import keras
from keras import backend as K
import tensorflow as tf

from AC_Agent_module import AC_Agent


# Play some random games at first to get a feel
def play_rand_games():
    env = gym.make('Pendulum-v0')
    max_steps = 50
    rand_episodes = 10
    rewards = []
    for _ in range(rand_episodes):
        env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            env.render()
            action = env.action_space.sample()

            new_state, reward, done, _ = env.step(action)
            # print(new_state, reward)

            episode_reward += reward

            if done:
                break

        rewards.append(episode_reward)

    print("Average score over time: " + str(sum(rewards) / rand_episodes))
    print(rewards)


# play_rand_games()


class Pendulum:
    def __init__(self):
        self.max_steps = 2000
        self.episodes = 10000
        self.env = gym.make('Pendulum-v0')
        self.env.reset()

        self.sess = tf.Session()
        K.get_session()

        self.actor_critic = AC_Agent(self.env, self.sess)

    def trainer(self, episodes=None, max_steps=None, render=False, verbose=False):
        if episodes is None:
            episodes = self.episodes
        if max_steps is None:
            max_steps = self.max_steps
        self.actor_critic.train_agent(episodes, max_steps, render, verbose)


if __name__ == "__main__":
    pendulum = Pendulum()
    import cProfile
    # cProfile.run('biped.train_walker(verbose=True)')
    pendulum.trainer(verbose=True)
