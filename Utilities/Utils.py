import matplotlib.pyplot as plt
import torch

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
    plt.savefig("all.png", dpi=300, bbox_inches='tight')

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")

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