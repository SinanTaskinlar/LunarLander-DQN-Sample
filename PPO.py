import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

class PPOModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers=(256, 256), clip_ratio=0.2):
        super(PPOModel, self).__init__()
        layers = []
        input_dim = state_size
        for layer_size in hidden_layers:
            layers.append(nn.Linear(input_dim, layer_size))
            layers.append(nn.ReLU())
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

def compute_advantage(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    dones = dones.float()
    deltas = rewards + gamma * next_values * (1 - dones) - values
    advantages = []
    advantage = 0
    for delta in deltas.flip(0):
        advantage = delta + gamma * lam * advantage
        advantages.insert(0, advantage)
    return torch.stack(advantages).float()

class PPOTrainer:
    def __init__(self, env, state_size, action_size, config):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU support
        self.config = config
        self.model = PPOModel(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.clip_ratio = self.model.clip_ratio

    def train(self, max_episodes, max_timesteps_per_episode=1000, eval_freq=100):
        rewards = []
        best_reward = -float('inf')
        for episode in range(max_episodes):
            state, _ = self.env.reset()
            state = torch.FloatTensor(state).to(self.device)
            total_reward = 0
            done = False
            values = []
            actions = []
            states = []
            rewards_list = []
            dones = []
            next_values = []

            while not done:
                policy, value = self.model(state)
                action = torch.distributions.Categorical(policy).sample().item()

                next_state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated
                next_state = torch.FloatTensor(next_state).to(self.device)

                values.append(value)
                actions.append(action)
                states.append(state)
                rewards_list.append(reward)
                dones.append(done)
                next_values.append(self.model(next_state)[1])

                state = next_state
                total_reward += reward

                if len(states) >= max_timesteps_per_episode:
                    break

            values = torch.stack(values)
            rewards_list = torch.tensor(rewards_list).to(self.device)
            next_values = torch.stack(next_values).to(self.device)
            dones = torch.tensor(dones).to(self.device)

            advantages = compute_advantage(rewards_list, values, next_values, dones, gamma=self.config['gamma'], lam=self.config['gae_lambda'])

            self._update_model(states, actions, advantages, values, rewards_list)

            rewards.append(total_reward)
            print(f"Episode {episode}, Reward: {total_reward}")

            if episode % eval_freq == 0:
                eval_reward = self.evaluate_model()
                print(f"Evaluation Reward: {eval_reward}")

            if total_reward > best_reward:
                best_reward = total_reward
            elif episode > 3000 and total_reward < best_reward * 0.9:
                print(f"Early stopping at episode {episode} due to performance drop.")
                break

            if episode % self.config.get('save_freq', 1000) == 0:
                torch.save(self.model.state_dict(), f"models/ppo/ppo_model_{episode}.pth")

        return rewards

    def _update_model(self, states, actions, advantages, values, rewards):
        states = torch.stack(states).float().to(self.device)  # Ensuring float32 precision
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        advantages = advantages.to(self.device)
        values = values.to(self.device)
        rewards = rewards.to(self.device).float()  # Explicitly setting rewards to float32

        old_policy, _ = self.model(states)
        old_log_probs = torch.log(old_policy.gather(1, actions.unsqueeze(1)).squeeze(1))

        policy, _ = self.model(states)
        log_probs = torch.log(policy.gather(1, actions.unsqueeze(1)).squeeze(1))
        ratio = torch.exp(log_probs - old_log_probs)
        obj = ratio * advantages
        clipped_obj = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(obj, clipped_obj).mean()

        _, new_value = self.model(states)
        value_loss = F.mse_loss(new_value.squeeze(), rewards)

        # Entropy loss
        entropy_loss = -(policy * torch.log(policy)).sum(dim=-1).mean()

        # Total loss with entropy regularization
        total_loss = policy_loss + 0.5 * value_loss - self.config['entropy_coeff'] * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
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

def plot_ppo(ppo_rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(ppo_rewards, label="PPO", color="green")
    plt.xlabel("Deneme Sayısı")
    plt.ylabel("Ödül Değeri")
    plt.title("LunarLander Ortamında PPO Algoritması Performansı")
    plt.legend()
    plt.grid()
    plt.show()

# Configuration with the suggested updates

