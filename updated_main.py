# Gerekli Kütüphaneleri Yükleme
import random
from collections import deque
import gymnasium as gym
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. LunarLander Ortamını Hazırlama
env_name = "LunarLander-v3"
env = gym.make(env_name)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

print(f"Durum Uzayı Boyutu: {state_size}")
print(f"Eylem Uzayı Boyutu: {action_size}")

# 2. DQN Modeli
class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, memory_size=2000):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=memory_size)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential([
            Dense(256, activation='relu', input_shape=(self.state_size,)),
            Dense(256, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))
        states = np.squeeze(states, axis=1)
        next_states = np.squeeze(next_states, axis=1)

        current_qs = self.model.predict(states, verbose=0)
        next_qs = self.target_model.predict(next_states, verbose=0)

        for i in range(batch_size):
            target = rewards[i] + (self.gamma * np.max(next_qs[i]) * (1 - dones[i]))
            current_qs[i][actions[i]] = target

        self.model.fit(states, current_qs, verbose=0, batch_size=batch_size)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# 3. A3C Modeli
class A3CAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.actor = self._build_actor_model()
        self.critic = self._build_critic_model()

    def _build_actor_model(self):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.state_size,)),
            Dense(self.action_size, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def _build_critic_model(self):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.state_size,)),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        policy = self.actor.predict(state, verbose=0)[0]
        return np.random.choice(self.action_size, p=policy)

    def learn(self, state, action, reward, next_state, done):
        state = np.reshape(state, [1, self.state_size])
        next_state = np.reshape(next_state, [1, self.state_size])
        target = reward + self.gamma * self.critic.predict(next_state, verbose=0) * (1 - done)

        # Critic Güncellemesi
        with tf.GradientTape() as tape:
            value = self.critic(state, training=True)
            critic_loss = tf.keras.losses.MSE(target, value)
        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

        # Actor Güncellemesi
        with tf.GradientTape() as tape:
            policy = self.actor(state, training=True)
            log_policy = tf.math.log(policy[0, action])
            advantage = target - value
            actor_loss = -log_policy * advantage
        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

# 4. PPO Modeli
class PPOAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, clip_epsilon=0.2):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.actor = self._build_actor_model()
        self.critic = self._build_critic_model()

    def _build_actor_model(self):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.state_size,)),
            Dense(self.action_size, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def _build_critic_model(self):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.state_size,)),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        policy = self.actor.predict(state, verbose=0)[0]
        return np.random.choice(self.action_size, p=policy)

    def learn(self, states, actions, rewards, next_states, dones, old_probs):
        # PPO öğrenme mantığı burada geliştirilebilir
        pass

# 5. Eğitim Fonksiyonu
def train_agent(agent_class, env, episodes=500, batch_size=32):
    agent = agent_class(state_size, action_size)
    episode_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward, done = 0, False

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

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{episodes}: Reward = {total_reward:.2f}")

    # Eğitim Sonuçları
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, label="Rewards")
    plt.title(f"Training Rewards ({agent_class.__name__})")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.show()

# 6. Ana Program
if __name__ == "__main__":
    train_agent(PPOAgent, env, episodes=200)