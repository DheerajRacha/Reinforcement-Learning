import os
import gym
import math
import random
import numpy as np
from itertools import count
from collections import namedtuple
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class AgentDDQN:
    def __init__(self, env, num_state, layers=None):
        if layers is None:
            layers = [16, 16]
        self.env_id = env
        self.env = gym.make(env)
        self.num_states = num_state
        self.layers = layers
        self.policy_net = self._build_model()
        self.target_net = self._build_model()

        self.memory_limit = None
        self.memory = None
        self.eps_start = None
        self.eps_end = None
        self.eps_decay = None
        self.steps_done = 0

    def _build_model(self):
        model = Sequential()

        model.add(Dense(self.layers[0], input_dim=self.num_states, activation='relu'))
        model.add(Dense(self.layers[1], activation='relu'))
        model.add(Dense(self.env.action_space.n, activation='linear'))

        model.compile(loss='mse', optimizer='adam')
        return model

    def select_action(self, state, mode="inference"):
        if mode == "inference":
            return np.argmax(self.policy_net.predict(np.array([state]))[0])
        else:
            sample = random.random()
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
                -1.0 * self.steps_done / self.eps_decay)
            self.steps_done += 1
            if sample > eps_threshold:
                return np.argmax(self.policy_net.predict(np.array([state]))[0])
            else:
                return self.env.action_space.sample()

    def _optimize_policy_net(self, batch_size=32, gamma=0.99):
        if len(self.memory) < batch_size:
            return

        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        curr_state_batch = np.array(batch.state)
        next_state_batch = np.array(batch.next_state)
        action_batch = np.array(batch.action)
        reward_batch = np.array(batch.reward)
        done_batch = np.array(batch.done)

        Y = []
        for i in range(batch_size):
            state, next_state, action, reward, done = curr_state_batch[i], next_state_batch[i], action_batch[i], \
                                                      reward_batch[i], done_batch[i]
            y = list(self.policy_net.predict(np.array([state]))[0])
            if done:
                y[action] = reward
            else:
                y_ = self.target_net.predict(np.array([next_state]))[0]
                action_index = np.argmax(self.policy_net.predict(np.array([next_state]))[0])
                y[action] = reward + gamma * y_[action_index]
            Y.append(y)

        self.policy_net.fit(curr_state_batch, np.array(Y), epochs=1, verbose=0)

    def train_agent(self, num_episodes=1500, target_update=3, memory_limit=50000, eps_start=0.9, eps_end=0.05,
                    eps_decay=200):
        self.memory_limit = memory_limit
        self.memory = ReplayMemory(memory_limit)
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0

        checkpoints_dir = "checkpoints/DDQN_{}".format(self.env_id.split("-")[0])
        if not os.path.isdir(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        checkpoints_path = checkpoints_dir + "/ckpt_{ep:04d}.h5"

        for episode in range(num_episodes):
            current_state = self.env.reset()

            total_reward = 0
            episode_duration = 0
            for _ in count():
                episode_duration += 1
                action = self.select_action(current_state, "train")
                next_state, reward, done, info = self.env.step(action)
                total_reward += reward

                if done:
                    if next_state[0] > 0.5:
                        reward = 1
                    else:
                        reward = -1

                self.memory.push(current_state, action, next_state, reward, done)
                current_state = next_state

                self._optimize_policy_net()
                if done:
                    break
            print("Episode: {}, Reward: {}, Duration: {}".format(episode, total_reward, episode_duration))

            if episode % target_update == 0:
                self.target_net.set_weights(self.policy_net.get_weights())
            if episode % 100 == 0:
                self.policy_net.save_weights(checkpoints_path.format(ep=episode))

    def load_from_checkpoints(self, path):
        self.policy_net.load_weights(path)

    def test_agent(self, checkpoints_path, num_episodes=10):
        self.load_from_checkpoints(checkpoints_path)

        for episode in range(num_episodes):
            state = self.env.reset()
            while True:
                action = np.argmax(self.policy_net.predict(state), axis=1)[0]
                state, r, done, _ = self.env.step(action)
                self.env.render()
                if done:
                    break
        self.env.close()
