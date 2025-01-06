import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PPOModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers=(256, 256), clip_ratio=0.2):
        super(PPOModel, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)
        self.policy_head = nn.Linear(input_dim, action_dim)
        self.value_head = nn.Linear(input_dim, 1)
        self.clip_ratio = clip_ratio

    def forward(self, x):
        shared_output = self.shared_layers(x)
        policy_logits = self.policy_head(shared_output)
        value = self.value_head(shared_output)
        return torch.softmax(policy_logits, dim=-1), value


class PPOTrainer:
    def __init__(self, env, state_dim, action_dim, config):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PPOModel(state_dim, action_dim, config['hidden_layers'], config['clip_ratio']).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])

        self.gamma = config['gamma']
        self.lam = config['gae_lambda']
        self.clip_ratio = config['clip_ratio']
        self.entropy_coeff = config['entropy_coeff']
        self.grad_clip = config.get('grad_clip', None)
        self.batch_size = config['batch_size']
        self.update_epochs = config.get('update_epochs', 10)
        self.reward_scaling = config.get('reward_scaling', 1.0)

    def train(self, max_episodes):
        episode_rewards = []
        for episode in range(max_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            done = False
            total_reward = 0
            states, actions, rewards, dones, values = [], [], [], [], []

            while not done:
                policy, value = self.model(state.unsqueeze(0))
                action_dist = torch.distributions.Categorical(policy)
                action = action_dist.sample()

                next_state, reward, done, truncated, _ = self.env.step(action.item())
                done = done or truncated

                rewards.append(reward * self.reward_scaling)

                states.append(state)
                actions.append(action)
                dones.append(done)
                values.append(value)

                state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
                total_reward += reward

            episode_rewards.append(total_reward)
            advantages, returns = self.compute_gae(rewards, dones, values)
            self.update_policy(states, actions, advantages, returns)

            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode}, Avg Reward: {avg_reward}")

        return episode_rewards

    def compute_gae(self, rewards, dones, values):
        values = torch.cat(values).squeeze(-1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * (1 - dones[t]) * (values[t + 1] if t + 1 < len(values) else 0) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.lam * (1 - dones[t]) * last_advantage

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update_policy(self, states, actions, advantages, returns):
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        advantages = advantages.detach()
        returns = returns.detach()

        for _ in range(self.update_epochs):
            policy, values = self.model(states)
            action_probs = policy.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            old_action_probs = action_probs.detach()

            ratio = action_probs / (old_action_probs + 1e-10)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            value_loss = nn.MSELoss()(values.squeeze(-1), returns)
            entropy_loss = -(policy * torch.log(policy + 1e-10)).sum(dim=-1).mean()

            loss = policy_loss + 0.5 * value_loss - self.entropy_coeff * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()


def plot_ppo(ppo_rewards):
    plt.figure(figsize=(10, 6))
    # Hareketli ortalama hesaplama
    window_size = 20
    smoothed_rewards = [sum(ppo_rewards[max(0, i - window_size):i]) / min(i + 1, window_size) for i in
                        range(len(ppo_rewards))]
    plt.plot(smoothed_rewards, label="PPO (smoothed)", color="green")
    plt.xlabel("Deneme Sayısı")
    plt.ylabel("Ödül Değeri")
    plt.title("LunarLander Ortamında PPO Algoritması Performansı")
    plt.legend()
    plt.grid()

    # Önce kaydet, sonra göster
    plt.savefig("~/Output/ppo/ppo_results.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()  # Belleği temizle


def PPOstart(environment_name="LunarLander-v3", render_mode=None, max_episodes=5000):
    ppo_env = gym.make(environment_name, render_mode=render_mode)
    ppo_config = {
        'lr': 2e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_ratio': 0.2,
        'entropy_coeff': 0.01,
        'grad_clip': 0.5,
        'batch_size': 64,
        'hidden_layers': (256, 256),
        'update_epochs': 5,
        'reward_scaling': 0.01
    }
    ppo_trainer = PPOTrainer(ppo_env, ppo_env.observation_space.shape[0], ppo_env.action_space.n, ppo_config)
    ppo_start_time = time.time()
    ppo_rewards = ppo_trainer.train(max_episodes=max_episodes)
    ppo_stop_time = time.time()
    plot_ppo(ppo_rewards)
    return ppo_rewards, ppo_stop_time - ppo_start_time