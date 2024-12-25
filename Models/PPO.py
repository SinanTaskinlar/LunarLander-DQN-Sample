import time
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np


class PPOModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers=(128, 128), clip_ratio=0.2):
        super(PPOModel, self).__init__()
        layers = []
        input_dim = state_size
        for layer_size in hidden_layers:
            layers.append(nn.Linear(input_dim, layer_size))
            layers.append(nn.LeakyReLU())
            input_dim = layer_size
        self.shared_layers = nn.Sequential(*layers)
        self.policy_head = nn.Linear(input_dim, action_size)
        self.value_head = nn.Linear(input_dim, 1)
        self.clip_ratio = clip_ratio

    def forward(self, x):
        shared = self.shared_layers(x)
        policy = torch.softmax(self.policy_head(shared), dim=-1)
        value = self.value_head(shared)
        return policy, value


class PPOTrainer:
    def __init__(self, env, state_size, action_size, config):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU support
        self.config = config
        self.model = PPOModel(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=100,
            verbose=True
        )
        self.clip_ratio = self.model.clip_ratio

        self.running_mean = torch.zeros(state_size).to(self.device)
        self.running_std = torch.ones(state_size).to(self.device)
        self.epsilon = 1e-8
        self.num_updates = 0

    def train(self, max_episodes, max_timesteps_per_episode=1000, eval_freq=100):
        best_reward = -float('inf')
        episode_rewards = []
        
        for episode in range(max_episodes):
            state, _ = self.env.reset()
            state = torch.FloatTensor(state).to(self.device)
            total_reward = 0
            done = False
            
            while not done:
                state = self.normalize_state(state)
                policy, value = self.model(state)
                action = torch.distributions.Categorical(policy).sample().item()

                next_state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated
                
                # Reward clipping ve scaling
                reward = np.clip(reward, -10, 10) * self.config['reward_scale']
                
                next_state = torch.FloatTensor(next_state).to(self.device)
                total_reward += reward

                # ... mevcut training kodu ...

                if total_reward >= self.config['early_stop_reward']:
                    print(f"Erken durma: Episode {episode} başarıyla tamamlandı!")
                    break

            episode_rewards.append(total_reward)
            
            # Her 100 episode'da bir ortalama ödülü yazdır
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode}, Ortalama Ödül: {avg_reward:.2f}")
                
                # En iyi modeli kaydet
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    torch.save(self.model.state_dict(), "best_lunarlander_model.pth")

        return episode_rewards

    def _update_model(self, states, actions, advantages, values, rewards):
        states = torch.stack(states).float().to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        advantages = advantages.to(self.device)
        values = values.to(self.device)
        rewards = rewards.to(self.device).float()
        batch_size = states.size(0)
        indices = torch.randperm(batch_size)

        for i in range(0, batch_size, self.config['batch_size']):
            batch_indices = indices[i:i + self.config['batch_size']]
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_values = values[batch_indices]
            batch_rewards = rewards[batch_indices]

            old_policy, _ = self.model(batch_states)
            old_log_probs = torch.log(old_policy.gather(1, batch_actions.unsqueeze(1)).squeeze(1))

            policy, _ = self.model(batch_states)
            log_probs = torch.log(policy.gather(1, batch_actions.unsqueeze(1)).squeeze(1))
            ratio = torch.exp(log_probs - old_log_probs)
            obj = ratio * batch_advantages
            clipped_obj = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
            policy_loss = -torch.min(obj, clipped_obj).mean()

            _, new_value = self.model(batch_states)
            value_pred_clipped = batch_values + torch.clamp(
                new_value.squeeze() - batch_values,
                -self.clip_ratio,
                self.clip_ratio
            )
            value_losses = f.mse_loss(new_value.squeeze(), batch_rewards)
            value_losses_clipped = f.mse_loss(value_pred_clipped, batch_rewards)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()

            entropy_loss = -(policy * torch.log(policy)).sum(dim=-1).mean()

            total_loss = policy_loss + 0.5 * value_loss - self.config['entropy_coeff'] * entropy_loss

            self.optimizer.zero_grad()
            total_loss.backward()

            if self.config.get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            self.optimizer.step()

    def evaluate_model(self, num_episodes=10):
        total_reward = 0
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            state = torch.FloatTensor(state).to(self.device)
            done = False
            while not done:
                policy, _ = self.model(state)
                action = torch.distributions.Categorical(policy).sample().item()
                state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated
                state = torch.FloatTensor(state).to(self.device)
                total_reward += reward
        return total_reward / num_episodes

    def normalize_state(self, state):
        return (state - self.running_mean) / (self.running_std + self.epsilon)

    def update_running_stats(self, states):
        states = torch.stack(states).to(self.device)
        batch_mean = torch.mean(states, dim=0)
        batch_std = torch.std(states, dim=0)

        # Daha yumuşak güncelleme için momentum ekleyelim
        momentum = 0.99
        self.running_mean = momentum * self.running_mean + (1-momentum) * batch_mean
        self.running_std = momentum * self.running_std + (1-momentum) * batch_std


def compute_advantage(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    dones = dones.float()
    deltas = rewards + gamma * next_values * (1 - dones) - values.squeeze()
    advantages = torch.zeros_like(deltas)
    advantage = 0
    
    # Avantajları geriye doğru hesapla
    for t in reversed(range(len(deltas))):
        advantage = deltas[t] + gamma * lam * advantage * (1 - dones[t])
        advantages[t] = advantage
    
    # Avantajları normalize et
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages


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
    plt.savefig("ppo_results.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()  # Belleği temizle


def PPOstart(environment_name="LunarLander-v3", render_mode=None, max_episodes=5000):
    ppo_env = gym.make(environment_name, render_mode=render_mode)
    ppo_config = {
        'lr': 3e-4,
        'gamma': 0.99,
        'save_freq': 1000,
        'clip_ratio': 0.2,
        'entropy_coeff': 0.01,
        'gae_lambda': 0.95,
        'grad_clip': 0.5,
        'batch_size': 64,
        'buffer_size': 2048,
        'num_epochs': 10,
        'target_kl': 0.01,
        # LunarLander için özel parametreler
        'reward_scale': 0.1,  # Ödülleri ölçeklendir
        'max_timesteps': 1000,  # Her episode için maksimum adım
        'early_stop_reward': 200  # Bu ödüle ulaşınca erken bitir
    }
    ppo_trainer = PPOTrainer(ppo_env, ppo_env.observation_space.shape[0], ppo_env.action_space.n, ppo_config)
    ppo_start_time = time.time()
    ppo_rewards = ppo_trainer.train(max_episodes=max_episodes)
    ppo_stop_time = time.time()
    plot_ppo(ppo_rewards)
    return ppo_rewards, ppo_stop_time - ppo_start_time