import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense


# Replay Buffer of previously performed actions.
# This is to allow for delayed learning.
class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        # max size.
        self.mem_size = max_size
        # first available memory position.
        self.mem_cntr = 0
        # Agent memory.
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        # Memory of the new states seen as a result of actions taken.
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        # Action memory
        self.action_memory = np.zeros((self.mem_size, n_actions))
        # Reward memory
        self.reward_memory = np.zeros(self.mem_size)
        # Array to keep track of the terminal flags received from the environment.
        # this is because the value of a terminal state is zero, so no reward should follow this.
        self.terminal_mermory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done ):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_mermory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_mermory[batch]

        return states, actions, rewards, states_, dones

