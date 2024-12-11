import random
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


# DQN Modeli
class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.005, gamma=0.95, epsilon=1, epsilon_decay=0.999, epsilon_min=0.1, memory_size=2000):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=memory_size)

        # Modeli ve hedef modelini oluştur
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = keras.Sequential([
            layers.InputLayer(shape=(self.state_size,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate)
        )
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.model.predict(state, verbose=0)[0])

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))

        states, next_states = np.squeeze(states, axis=1), np.squeeze(next_states, axis=1)
        current_qs = self.model.predict(states, verbose=0)
        next_qs = self.target_model.predict(next_states, verbose=0)

        for i in range(batch_size):
            target = rewards[i] + (self.gamma * np.max(next_qs[i]) * (1 - dones[i]))
            current_qs[i][actions[i]] = target

        self.model.fit(states, current_qs, verbose=0, batch_size=batch_size)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# A3C Modeli
class A3CAgent:
    def __init__(self, state_size, action_size, gamma=0.99, learning_rate=0.0007):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.actor = self._build_actor_model()
        self.critic = self._build_critic_model()

    def _build_actor_model(self):
        model = keras.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(self.action_size, activation="softmax")
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def _build_critic_model(self):
        model = keras.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="linear")
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        policy = self.actor.predict(state, verbose=0)[0]
        return np.random.choice(self.action_size, p=policy)

    def learn(self, state, action, reward, next_state, done):
        state = np.reshape(state, [1, self.state_size])
        next_state = np.reshape(next_state, [1, self.state_size])
        target = reward + self.gamma * self.critic.predict(next_state, verbose=0) * (1 - done)

        # Critic update
        with tf.GradientTape() as tape:
            value = self.critic(state, training=True)
            critic_loss = tf.keras.losses.MSE(target, value)
        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

        # Actor update
        with tf.GradientTape() as tape:
            policy = self.actor(state, training=True)
            log_policy = tf.math.log(policy[0, action])
            advantage = target - value
            actor_loss = -log_policy * advantage
        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))


# PPO Modeli
class PPOAgent:
    def __init__(self, state_size, action_size, gamma=0.99, learning_rate=0.0003, clip_epsilon=0.2):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.clip_epsilon = clip_epsilon
        self.actor = self._build_actor_model()
        self.critic = self._build_critic_model()

    def _build_actor_model(self):
        model = keras.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(self.action_size, activation="softmax")
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def _build_critic_model(self):
        model = keras.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="linear")
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        policy = self.actor.predict(state, verbose=0)[0]
        return np.random.choice(self.action_size, p=policy)

    def learn(self, states, actions, rewards, next_states, dones, old_probs):
        pass  # PPO detaylı uygulama.


# Ortak Eğitim Fonksiyonu
def train_agent(agent_class, env_name="LunarLander-v3", episodes=500, batch_size=32, render_mode=None):
    env = gym.make(env_name, render_mode=render_mode)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = agent_class(state_size, action_size)

    episode_rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward, done, step = 0, False, 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            if isinstance(agent, DQNAgent):
                agent.remember(state, action, reward, next_state, done)
                agent.replay(batch_size)
            elif isinstance(agent, A3CAgent):
                agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            step += 1

        episode_rewards.append(total_reward)
        print(f"Episode: {episode}, Reward: {total_reward:.2f}")

    plt.plot(episode_rewards)
    plt.title(f"Training Rewards ({agent_class.__name__})")
    plt.show()
    env.close()


# Çalıştırma
if __name__ == "__main__":
    train_agent(PPOAgent, episodes=50, render_mode="human")
