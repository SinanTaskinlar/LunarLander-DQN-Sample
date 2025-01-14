import matplotlib.pyplot as plt
import torch


def plot_all_algorithms(dqn_rewards, ppo_rewards, a3c_rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(dqn_rewards, label="DQN", color="blue")
    plt.plot(ppo_rewards, label="PPO", color="red")
    plt.plot(a3c_rewards, label="A3C", color="gray")
    plt.xlabel("Deneme Sayısı")
    plt.ylabel("Ödül Değeri")
    plt.title("LunarLander Ortamında DQN-PPO-A3C Algoritma Karşılaştırması")
    plt.legend()
    plt.grid()
    plt.show()


def print_all_times(dqn_time, a3c_time, ppo_time):
    print(f"DQN time: {dqn_time:.1f} seconds")
    print(f"A3C time: {a3c_time:.1f} seconds")
    print(f"PPO time: {ppo_time:.1f} seconds")


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

    @staticmethod
    def log(episode, reward, **kwargs):
        log_data = {"Episode": episode, "Reward": reward}
        log_data.update(kwargs)

    def plot_rewards(self, rewards):
        print(f"Plotting rewards for {self.algorithm_name}.")
