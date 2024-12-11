#import os
import random
from collections import deque
from multiprocessing import Process
from multiprocessing import Manager
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

def plot_all_algorithms(dqn_rewards, ppo_rewards, a3c_rewards):

    plt.figure(figsize=(10, 6))
    plt.plot(dqn_rewards, label="DQN", color="blue")
    plt.plot(ppo_rewards, label="PPO", color="green")
    plt.plot(a3c_rewards, label="A3C", color="yellow")
    plt.xlabel("Deneme Sayısı")
    plt.ylabel("Ödül Değeri")
    plt.title("LunarLander Ortamında DQN-PPO-A3C Algoritma Karşılaştırması")
    plt.legend()
    plt.grid()
    plt.show()

# Utility class for wandb logging and reward plotting
class PlotAndLog:
    def __init__(self, algorithm_name):
        self.algorithm_name = algorithm_name
        # wandb.init(project="RL-Algorithms", name=self.algorithm_name, config={})

    @staticmethod
    def log(episode, reward, **kwargs):
        log_data = {"Episode": episode, "Reward": reward}
        log_data.update(kwargs)
        # wandb.log(log_data)

    def plot_rewards(self, rewards):
        print(f"Plotting rewards for {self.algorithm_name}.")

# Define DQN model
class DQNModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers=(64, 64)):
        super().__init__()
        layers = []
        input_dim = state_size
        for layer_size in hidden_layers:
            layers.append(nn.Linear(input_dim, layer_size))
            layers.append(nn.ReLU())
            input_dim = layer_size
        layers.append(nn.Linear(input_dim, action_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Trainer for DQN
class DQNTrainer:
    def __init__(self, env, state_size, action_size, config):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.model = DQNModel(state_size, action_size).to(self.device)
        self.target_model = DQNModel(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.memory = deque(maxlen=config.get('memory_size', 10000))
        self.epsilon = config.get('epsilon_start', 1.0)
        self.plot_and_log = PlotAndLog("DQN")

    def train(self, max_episodes=1000):
        rewards = []

        for episode in range(max_episodes):
            state, _ = self.env.reset()
            state = torch.FloatTensor(state).to(self.device)
            total_reward = 0
            done = False

            while not done:
                action = self._select_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated
                next_state = torch.FloatTensor(next_state).to(self.device)
                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward

                if len(self.memory) >= self.config['batch_size']:
                    self._learn()

            self.epsilon = max(
                self.config.get('epsilon_end', 0.01),
                self.epsilon * self.config.get('epsilon_decay', 0.995)
            )

            rewards.append(total_reward)
            #self.plot_and_log.log(episode, total_reward)
            print(f"Episode {episode}, Reward: {total_reward}")

            if episode % self.config.get('target_update_freq', 10) == 0:
                self.target_model.load_state_dict(self.model.state_dict())

        self.plot_and_log.plot_rewards(rewards)
        return rewards

    def _select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self.model(state)
                return torch.argmax(q_values).item()

    def _learn(self):
        batch = random.sample(self.memory, self.config['batch_size'])
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        rewards = torch.FloatTensor(rewards).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_model(next_states).max(1)[0]
        target_q = rewards + self.config['gamma'] * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
        self.plot_and_log = PlotAndLog("PPO")

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
            print(f"Episode {episode}, Reward: {total_reward}")

        self.plot_and_log.plot_rewards(rewards)
        return rewards

# Define A3C Model
class A3CModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers=(128, 128)):
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

# Worker for A3C training
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

    def train(self):
        for episode in range(self.config['max_episodes']):
            state, _ = self.env.reset()
            state = torch.FloatTensor(state).to(self.device)
            done = False
            total_reward = 0
            log_probs = []
            values = []
            rewards = []

            while not done:
                policy, value = self.local_model(state)
                action_dist = Categorical(policy)
                action = action_dist.sample()

                log_probs.append(action_dist.log_prob(action))
                values.append(value)

                next_state, reward, done, truncated, _ = self.env.step(action.item())
                done = done or truncated
                rewards.append(reward)
                total_reward += reward
                state = torch.FloatTensor(next_state).to(self.device)

                if done:
                    break

            # Store rewards for this worker in shared reward list
            self.reward_list.append(total_reward)

            # Compute advantages and update global model
            self._update_global_model(rewards, log_probs, values)

            print(f"Worker {self.worker_id} | Episode {episode} | Reward: {total_reward}")

    def _update_global_model(self, rewards, log_probs, values):
        # Compute returns and advantages
        returns = []
        g = 0
        for r in reversed(rewards):
            g = r + self.config['gamma'] * g
            returns.insert(0, g)

        returns = torch.FloatTensor(returns).to(self.device)
        log_probs = torch.stack(log_probs)
        values = torch.cat(values)

        advantages = returns - values.detach()

        # Policy loss and value loss
        policy_loss = -(log_probs * advantages).mean()
        value_loss = nn.MSELoss()(values, returns)

        # Backpropagation
        loss = policy_loss + self.config['value_loss_coef'] * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        for global_param, local_param in zip(self.global_model.parameters(), self.local_model.parameters()):
            global_param._grad = local_param.grad
        self.optimizer.step()

        # Synchronize local model with global model
        self.local_model.load_state_dict(self.global_model.state_dict())

# Main A3C Training Process
class A3CTrainer:
    def __init__(self, env, state_size, action_size, config):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.global_model = A3CModel(state_size, action_size).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.global_model.share_memory()
        self.optimizer = optim.Adam(self.global_model.parameters(), lr=config['lr'])

    def train(self):
        manager = Manager()
        reward_list = manager.list()  # Paylaşımlı ödül listesi. Plotting için bu şekilde yapıldı
        processes = []
        for worker_id in range(self.config['num_workers']):
            worker = A3CWorker(self.global_model, self.optimizer, self.env, self.config, worker_id, reward_list)
            process = Process(target=worker.train)
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        return reward_list

# Main function
def main():

    max_episodes = 5
    environment_name = "LunarLander-v3"
    render_mode = "human"
    # render_mode = None

    # A3C Configuration
    a3c_env = gym.make(environment_name, render_mode=render_mode)
    a3c_config = {
        'lr': 1e-4, #0.0001
        'gamma': 0.99,
        'value_loss_coef': 0.5,
        'num_workers': 4,
        'max_episodes': max_episodes
    }
    a3c_trainer = A3CTrainer(environment_name, a3c_env.observation_space.shape[0], a3c_env.action_space.n, a3c_config)
    a3c_rewards = a3c_trainer.train()

    # DQN Configuration
    dqn_env = gym.make(environment_name, render_mode=render_mode)
    dqn_config = {
        'lr': 1e-3, #0.0001
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay': 0.995,
        'batch_size': 128,
        'memory_size': 10000,
        'target_update_freq': 10
    }
    dqn_trainer = DQNTrainer(dqn_env, dqn_env.observation_space.shape[0], dqn_env.action_space.n, dqn_config)
    dqn_rewards = dqn_trainer.train(max_episodes=max_episodes)

    # PPO Configuration
    ppo_env = gym.make(environment_name, render_mode=render_mode)
    ppo_config = {
        'lr': 3e-4, #0.0003
        'gamma': 0.99
    }
    ppo_trainer = PPOTrainer(ppo_env, ppo_env.observation_space.shape[0], ppo_env.action_space.n, ppo_config)
    ppo_rewards = ppo_trainer.train(max_episodes=max_episodes)

    # Plot all algorithms
    plot_all_algorithms(dqn_rewards, ppo_rewards, a3c_rewards)

if __name__ == "__main__":
    main()
