import gymnasium as gym
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random
import matplotlib.pyplot as plt

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.memory = deque(maxlen=2000)

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = keras.Sequential([
            layers.InputLayer(input_shape=(self.state_size,)),
            layers.Dense(8, activation='relu'),
            layers.Dense(8, activation='relu'),
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

        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        states = np.array([x[0][0] for x in minibatch])
        next_states = np.array([x[3][0] for x in minibatch])

        current_q_values = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)

        X = []
        y = []

        for index, (state, action, reward, next_state, done) in enumerate(minibatch):
            if not done:
                max_next_q = np.max(next_q_values[index])
                new_q = reward + self.gamma * max_next_q
            else:
                new_q = reward

            current_q = current_q_values[index]
            current_q[action] = new_q

            X.append(state[0])
            y.append(current_q)

        self.model.fit(np.array(X), np.array(y), batch_size=batch_size, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_lunar_lander_dqn(episodes=500, render_mode=None):
    # Ortamı oluştur
    env = gym.make('LunarLander-v3', render_mode=render_mode)

    # Random seed değeri
    env.reset(seed=42)
    np.random.seed(42)
    random.seed(42)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    episode_rewards = []


    for episode in range(episodes):
        # Ortam reset
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])

        total_reward = 0
        done = False
        truncated = False
        step = 0

        while not (done or truncated):
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)

            next_state = np.reshape(next_state, [1, state_size])

            # Ödülü ve state kaydet
            agent.remember(state, action, reward, next_state, done or truncated)

            # update state
            state = next_state
            total_reward += reward
            step += 1

            agent.replay(32)

            if step > 1000:
                break

        episode_rewards.append(total_reward)

        if episode % 2 == 0:
            print(f"Episode: {episode}, Toplam Ödül: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    # Performans visualize
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title('Eğitim Boyunca Toplam Ödüller')
    plt.xlabel('Episode')
    plt.ylabel('Toplam Ödül')
    plt.tight_layout()
    plt.show()

    env.close()

# start trainn
if __name__ == "__main__":
    # None, 'human', 'rgb_array'
    train_lunar_lander_dqn(episodes=5, render_mode='human')
