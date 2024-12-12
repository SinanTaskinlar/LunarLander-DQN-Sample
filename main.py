# import os
import gymnasium as gym
import A3C
import DQN
import PPO
import Utils

# Main function
def main():
    max_episodes = 10000
    environment_name = "LunarLander-v3"
    # render_mode = "human"
    render_mode = None

    dqn_rewards = DQNstart(environment_name, render_mode, max_episodes)
    # a3c_rewards = A3Cstart(environment_name, render_mode, max_episodes)
    # ppo_rewards = PPOstart(environment_name, render_mode, max_episodes)
    #
    # Utils.plot_all_algorithms(dqn_rewards, ppo_rewards, a3c_rewards)

    # Kaydedilen modeli istersek yüklemek

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # dqn_model = DQNModel(a3c_env.observation_space.shape[0], a3c_env.action_space.n).to(device)
    # load_model(dqn_model, "dqn_model_500.pth")
    # ppo_model = PPOModel(a3c_env.observation_space.shape[0], a3c_env.action_space.n).to(device)
    # load_model(ppo_model, "ppo_model_500.pth")
    # a3c_model = A3CModel(a3c_env.observation_space.shape[0], a3c_env.action_space.n).to(device)
    # load_model(a3c_model, "a3c_model.pth")

    # Plot all algorithms

def DQNstart(environment_name, render_mode, max_episodes):
    # DQN Configuration
    dqn_env = gym.make(environment_name, render_mode=render_mode)
    dqn_config = {
        'lr': 1e-4,  # 0.0001
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay': 0.9995,
        'batch_size': 32,
        'memory_size': 10000,
        'target_update_freq': 30
    }
    dqn_trainer = DQN.DQNTrainer(dqn_env, dqn_env.observation_space.shape[0], dqn_env.action_space.n, dqn_config)
    dqn_rewards = dqn_trainer.train(max_episodes=max_episodes)
    DQN.plot_dqn(dqn_rewards)
    return dqn_rewards

def A3Cstart(environment_name, render_mode, max_episodes):
    # A3C Configuration
    a3c_env = gym.make(environment_name, render_mode=render_mode)
    a3c_config = {
        'lr': 1e-4,  # 0.0001
        'gamma': 0.99,
        'value_loss_coef': 0.5,
        'num_workers': 4,
        'max_episodes': int(max_episodes / 4)
    }
    a3c_trainer = A3C.A3CTrainer(environment_name, a3c_env.observation_space.shape[0], a3c_env.action_space.n,
                                 a3c_config)
    a3c_rewards = a3c_trainer.train()
    DQN.plot_dqn(a3c_rewards)
    return a3c_rewards

def PPOstart(environment_name, render_mode, max_episodes):
    # PPO Configuration
    ppo_env = gym.make(environment_name, render_mode=render_mode)
    ppo_config = {
        'lr': 3e-4,  # 0.0003
        'gamma': 0.99
    }
    ppo_trainer = PPO.PPOTrainer(ppo_env, ppo_env.observation_space.shape[0], ppo_env.action_space.n, ppo_config)
    ppo_rewards = ppo_trainer.train(max_episodes=max_episodes)
    PPO.plot_ppo(ppo_rewards)
    return ppo_rewards

if __name__ == "__main__":
    main()