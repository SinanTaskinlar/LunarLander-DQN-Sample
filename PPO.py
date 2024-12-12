# import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

import Utils


def plot_ppo(ppo_rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(ppo_rewards, label="DQN", color="blue")
    plt.xlabel("Deneme Sayısı")
    plt.ylabel("Ödül Değeri")
    plt.title("LunarLander Ortamında DQN-PPO-A3C Algoritma Karşılaştırması")
    plt.legend()
    plt.grid()
    plt.show()

# Define PPO model
class PPOModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers=(64, 64)):
        super().__init__()
        layers = []
        input_dim = state_size
        for layer_size in hidden_layers:
            layers.append(nn.Linear(input_dim, layer_size))
            layers.append(nn.ReLU())
            input_dim = layer_size
        self.shared_layers = nn.Sequential(*layers)
        self.policy_head = nn.Linear(input_dim, action_size)
        self.value_head = nn.Linear(input_dim, 1)

    def forward(self, x):
        shared = self.shared_layers(x)
        policy = torch.softmax(self.policy_head(shared), dim=-1)
        value = self.value_head(shared)
        return policy, value


# Trainer for PPO
class PPOTrainer:
    def __init__(self, env, state_size, action_size, config):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.model = PPOModel(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.plot_and_log = Utils.PlotAndLog("PPO")

    def train(self, max_episodes=1000):
        rewards = []

        for episode in range(max_episodes):
            state, _ = self.env.reset()
            state = torch.FloatTensor(state).to(self.device)
            total_reward = 0
            done = False

            while not done:
                policy, value = self.model(state)
                action = torch.distributions.Categorical(policy).sample().item()
                next_state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated
                next_state = torch.FloatTensor(next_state).to(self.device)
                total_reward += reward
                state = next_state

            rewards.append(total_reward)
            # self.plot_and_log.log(episode, total_reward)

            if episode % self.config.get('save_freq', 1000) == 0:
                Utils.save_model(self.model, f"models/ppo/ppo_model_{episode}.pth")

            print(f"Episode {episode}, Reward: {total_reward}")

        self.plot_and_log.plot_rewards(rewards)
        return rewards