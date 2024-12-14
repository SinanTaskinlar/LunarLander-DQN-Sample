import gymnasium as gym
import A3C
import DQN
import PPO
import Utils
import torch
import time




def main():
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDNN version: {torch.backends.cudnn.version()}')
    print(f'Device: {torch.cuda.get_device_name()}')

    max_episodes = 5000
    environment_name = "LunarLander-v3"
    # render_mode = "human"
    render_mode = None

    dqn_rewards, dqn_time = DQNstart(environment_name, render_mode, max_episodes)
    print(f"DQN time: {dqn_time}")

    a3c_rewards, a3c_time = A3Cstart(environment_name, render_mode, max_episodes)
    print(f"A3C time: {a3c_time}")

    ppo_rewards, ppo_time = PPOstart(environment_name, render_mode, max_episodes)
    print(f"PPO time: {ppo_time}")

    # Utils.plot_all_algorithms(dqn_rewards, ppo_rewards, a3c_rewards)

    # Kaydedilen modeli yüklemek

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # dqn_model = DQNModel(a3c_env.observation_space.shape[0], a3c_env.action_space.n).to(device)
    # load_model(dqn_model, "dqn_model_500.pth")
    # ppo_model = PPOModel(a3c_env.observation_space.shape[0], a3c_env.action_space.n).to(device)
    # load_model(ppo_model, "ppo_model_500.pth")
    # a3c_model = A3CModel(a3c_env.observation_space.shape[0], a3c_env.action_space.n).to(device)
    # load_model(a3c_model, "a3c_model.pth")

def DQNstart(environment_name, render_mode, max_episodes):

    dqn_env = gym.make(environment_name, render_mode=render_mode)
    dqn_config = {
        'lr': 1e-4,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay': 0.9995,
        'batch_size': 32,
        'memory_size': 10000,
        'target_update_freq': 30
    }
    dqn_trainer = DQN.DQNTrainer(dqn_env, dqn_env.observation_space.shape[0], dqn_env.action_space.n, dqn_config)
    dqn_start_time = time.time()
    dqn_rewards = dqn_trainer.train(max_episodes=max_episodes)
    dqn_stop_time = time.time()
    DQN.plot_dqn(dqn_rewards)
    return dqn_rewards, dqn_stop_time - dqn_start_time

def A3Cstart(environment_name, render_mode, max_episodes):
    a3c_env = gym.make(environment_name, render_mode=render_mode)
    a3c_config = {
        'lr': 1e-4,
        'gamma': 0.99,
        'value_loss_coef': 0.5,
        'num_workers': 4,
        'max_episodes': int(max_episodes / 4)
    }
    a3c_trainer = A3C.A3CTrainer(environment_name, a3c_env.observation_space.shape[0], a3c_env.action_space.n,
                                 a3c_config)
    a3c_start_time = time.time()
    a3c_rewards = a3c_trainer.train()
    a3c_stop_time = time.time()

    A3C.plot_a3c(a3c_rewards)
    return a3c_rewards, a3c_stop_time - a3c_start_time

def PPOstart(environment_name, render_mode, max_episodes):
    ppo_env = gym.make(environment_name, render_mode=render_mode)
    ppo_config = {
        'lr': 2e-4,  # Slightly reduced learning rate
        'gamma': 0.95,  # Reduced gamma for quicker reward optimization
        'save_freq': 1000,  # Model saving frequency
        'clip_ratio': 0.2,  # Clip ratio (can experiment with 0.1-0.3)
        'entropy_coeff': 0.01,  # Entropy term coefficient for exploration
        'gae_lambda': 0.95,  # Lambda for GAE (Generalized Advantage Estimation)
        'grad_clip': 0.5,  # Gradient clipping norm
    }
    ppo_trainer = PPO.PPOTrainer(ppo_env, ppo_env.observation_space.shape[0], ppo_env.action_space.n, ppo_config)
    ppo_start_time = time.time()
    ppo_rewards = ppo_trainer.train(max_episodes=max_episodes)
    ppo_stop_time = time.time()
    PPO.plot_ppo(ppo_rewards)
    return ppo_rewards, ppo_stop_time - ppo_start_time

if __name__ == "__main__":
    main()