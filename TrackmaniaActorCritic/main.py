import pybullet_envs
import gym
import tensorflow as tf
import tensorflow.keras.models
import numpy as np
from sac_tf2 import Agent
from utils import plot_learning_curve
from gym import wrappers
from environment import TrackmaniaEnv

if __name__ == '__main__':

    VAEncoder = tf.keras.models.load_model('CVAE/Models/model', compile=True)
    #VAEncoder.summary()

    agent = Agent(input_dims=VAEncoder.get_layer(index=VAEncoder.layers[-1:]).shape(), max_action=1, n_actions=2)
    n_games = 250

    env = TrackmaniaEnv()

    best_score = 0
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            actions = agent.choose_action(observation)
            observation_, reward, done, info = env.step(actions)
            score += reward
            agent.remember(observation, actions, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)