import time
from multiprocessing import Manager, Process
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from Utilities import Utils

def smooth_rewards(rewards, window=50):
    return np.convolve(rewards, np.ones(window) / window, mode='valid')

def plot_a3c(a3c_rewards):
    smoothed = smooth_rewards(a3c_rewards)
    plt.figure(figsize=(10, 6))
    plt.plot(a3c_rewards, label="A3C Raw", color="yellow", alpha=0.5)
    plt.plot(smoothed, label="Smoothed", color="blue")
    plt.xlabel("Deneme Sayısı")
    plt.ylabel("Ödül Değeri")
    plt.title("LunarLander Ortamında A3C Algoritması Performansı")
    plt.legend()
    plt.grid()
    plt.savefig("a3c_smoothed.png", dpi=300, bbox_inches='tight')
    plt.show()

class A3CModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers=(256, 128)):
        super().__init__()
        layers = []
        input_dim = state_size
        for layer_size in hidden_layers:
            layers.append(nn.Linear(input_dim, layer_size))
            layers.append(nn.LeakyReLU())
            input_dim = layer_size
        self.shared_layers = nn.Sequential(*layers)
        self.policy_head = nn.Linear(input_dim, action_size)
        self.value_head = nn.Linear(input_dim, 1)

    def forward(self, x):
        shared = self.shared_layers(x)
        policy = torch.softmax(self.policy_head(shared), dim=-1)
        value = self.value_head(shared)
        return policy, value

class A3CWorker:
    def __init__(self, global_model, optimizer, env_name, config, worker_id, reward_list):
        self.global_model = global_model
        self.optimizer = optimizer
        self.env = gym.make(env_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.worker_id = worker_id
        self.local_model = A3CModel(
            self.env.observation_space.shape[0],
            self.env.action_space.n
        ).to(self.device)
        self.reward_list = reward_list
        self.memory_size = 1000
        self.memory = []
        self.reward_scaling = config.get('reward_scaling', 0.1)

    def _store_transition(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        for episode in range(self.config['max_episodes']):
            state, _ = self.env.reset()
            state = torch.FloatTensor(state).to(self.device)
            done = False
            total_reward = 0
            log_probs = []
            values = []
            rewards = []
            states = []

            while not done:
                policy, value = self.local_model(state)
                action_dist = Categorical(policy)
                action = action_dist.sample()

                log_probs.append(action_dist.log_prob(action))
                values.append(value)
                states.append(state)

                next_state, reward, done, truncated, _ = self.env.step(action.item())
                done = done or truncated

                # Reward scaling ve clipping
                reward *= self.reward_scaling
                reward = max(min(reward, 1), -1)

                rewards.append(reward)
                total_reward += reward
                state = torch.FloatTensor(next_state).to(self.device)

                if done:
                    break

            self.reward_list.append(total_reward)
            self._update_global_model(rewards, log_probs, values, states, done)

            if episode % 30 == 0:
                print(f"Worker {self.worker_id} | Episode {episode} | Reward: {total_reward}")

    def _update_global_model(self, rewards, log_probs, values, states, done):

        states = torch.stack(states).to(self.device)
        values = torch.cat(values).squeeze(-1).to(self.device)

        with torch.no_grad():
            if done:
                next_value = torch.tensor(0.0, device=self.device)
            else:
                next_value = self.local_model(states[-1].unsqueeze(0))[1].squeeze(-1)

        returns = self._compute_gae(rewards, values, next_value, done)
        returns = torch.FloatTensor(returns).to(self.device)
        log_probs = torch.stack(log_probs).to(self.device)
        advantages = returns - values.detach()

        # Entropy loss for exploration
        entropy_loss = -torch.sum(torch.log(log_probs.exp() + 1e-10) * log_probs.exp(), dim=0).mean()

        policy_loss = -(log_probs * advantages).mean()
        value_loss = nn.MSELoss()(values, returns)

        # Total loss
        loss = policy_loss + self.config['value_loss_coef'] * value_loss - self.config['entropy_coef'] * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        for global_param, local_param in zip(self.global_model.parameters(), self.local_model.parameters()):
            global_param._grad = local_param.grad
        torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=0.5)
        self.optimizer.step()

        self.local_model.load_state_dict(self.global_model.state_dict())

    def _compute_gae(self, rewards, values, next_value, done):
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_value = 0 if done else next_value
            else:
                next_value = values[step + 1]

            delta = rewards[step] + self.config['gamma'] * next_value - values[step]
            gae = delta + self.config['gamma'] * self.config['gae_lambda'] * gae
            returns.insert(0, gae + values[step])

        return returns

class A3CTrainer:
    def __init__(self, env, state_size, action_size, config):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.global_model = A3CModel(state_size, action_size).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.global_model.share_memory()
        self.optimizer = optim.Adam(self.global_model.parameters(), lr=config['lr'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                              T_max=config['max_episodes'],
                                                              eta_min=0)

    def train(self):
        manager = Manager()
        reward_list = manager.list()
        processes = []
        for worker_id in range(self.config['num_workers']):
            worker = A3CWorker(self.global_model, self.optimizer, self.env, self.config, worker_id, reward_list)
            process = Process(target=worker.train)
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        Utils.save_model(self.global_model, f"SavedModels/a3c/a3c_model.pth")
        print("A3C model saved.")

        return list(reward_list)

def A3Cstart(environment_name="LunarLander-v3", render_mode=None, max_episodes=5000):
    a3c_env = gym.make(environment_name, render_mode=render_mode)
    a3c_config = {
        'lr': 1e-4,
        'gamma': 0.99,
        'value_loss_coef': 0.25,
        'num_workers': 8,
        'max_episodes': int(max_episodes / 8),
        'entropy_coef': 0.005,
        'max_grad_norm': 0.5,
        'gae_lambda': 0.95,
        'reward_scaling': 0.1
    }
    a3c_trainer = A3CTrainer(environment_name, a3c_env.observation_space.shape[0], a3c_env.action_space.n,
                             a3c_config)
    a3c_start_time = time.time()
    a3c_rewards = a3c_trainer.train()
    a3c_stop_time = time.time()

    plot_a3c(a3c_rewards)
    return a3c_rewards, a3c_stop_time - a3c_start_time