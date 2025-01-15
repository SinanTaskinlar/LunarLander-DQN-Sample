import gymnasium as gym
import optuna
import torch

from Models import A3C, DQN, PPO

env_name = "LunarLander-v3"
# render_mode= "human"
render_mode = None
n_trails = 10
max_episodes = 50


def objective_dqn(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.99, 0.9999)
    batch_size = trial.suggest_int("batch_size", 16, 128)
    target_update_freq = trial.suggest_int("target_update_freq", 10, 50)

    dqn_config = {
        'lr': lr,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay': epsilon_decay,
        'batch_size': batch_size,
        'memory_size': 10000,
        'target_update_freq': target_update_freq
    }

    dqn_env = gym.make(env_name, render_mode)
    dqn_trainer = DQN.DQNTrainer(dqn_env, dqn_env.observation_space.shape[0], dqn_env.action_space.n, dqn_config)
    dqn_rewards = dqn_trainer.train(max_episodes=max_episodes)
    avg_reward = sum(dqn_rewards[-10:]) / 10
    return avg_reward


def objective_a3c(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    entropy_coef = trial.suggest_float("entropy_coef", 0.001, 0.01)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    reward_scaling = trial.suggest_float("reward_scaling", 0.01, 0.2)
    a3c_config = {
        'lr': lr,
        'gamma': 0.99,
        'value_loss_coef': 0.25,
        'num_workers': 8,
        'max_episodes': max_episodes,
        'entropy_coef': entropy_coef,
        'max_grad_norm': 0.5,
        'gae_lambda': gae_lambda,
        'reward_scaling': reward_scaling
    }

    a3c_env = gym.make(env_name, render_mode)
    a3c_trainer = A3C.A3CTrainer(env_name, a3c_env.observation_space.shape[0], a3c_env.action_space.n,
                                 a3c_config)
    a3c_rewards = a3c_trainer.train()
    avg_reward = sum(a3c_rewards[-10:]) / 10
    return avg_reward


def objective_ppo(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    clip_ratio = trial.suggest_float("clip_ratio", 0.1, 0.3)
    entropy_coeff = trial.suggest_float("entropy_coeff", 0.005, 0.05)
    batch_size = trial.suggest_int("batch_size", 32, 128)
    update_epochs = trial.suggest_int("update_epochs", 5, 15)
    reward_scaling = trial.suggest_float("reward_scaling", 0.001, 0.05)
    ppo_config = {
        'lr': lr,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_ratio': clip_ratio,
        'entropy_coeff': entropy_coeff,
        'grad_clip': 0.5,
        'batch_size': batch_size,
        'hidden_layers': (256, 256),
        'update_epochs': update_epochs,
        'reward_scaling': reward_scaling
    }

    ppo_env = gym.make(env_name, render_mode)
    ppo_trainer = PPO.PPOTrainer(ppo_env, ppo_env.observation_space.shape[0], ppo_env.action_space.n,
                                 ppo_config)
    ppo_rewards = ppo_trainer.train(max_episodes)
    avg_reward = sum(ppo_rewards[-10:]) / 10
    return avg_reward


def start_optimization():
    print(f'PyTorch version: {torch.__version__}')
    cudnn_version = torch.backends.cudnn.version()
    print(f'CUDNN version: {cudnn_version if cudnn_version else "None"}')

    if torch.cuda.is_available():
        print(f'Device: {torch.cuda.get_device_name(0)}')
    else:
        print("CUDA desteklenmiyor, CPU kullanılacak.")

    # DQN optimizasyon çalışmasını başlat
    study_dqn = optuna.create_study(direction="maximize")
    study_dqn.optimize(objective_dqn, n_trials=n_trails)

    # En iyi parametreleri ve değeri yazdır
    print("DQN Optimizasyon Bitti")
    print("Best value:", study_dqn.best_value)
    print("Best params:", study_dqn.best_params)

    # A3C optimizasyon çalışmasını başlat
    study_a3c = optuna.create_study(direction="maximize")
    study_a3c.optimize(objective_a3c, n_trials=n_trails)

    print("A3C Optimizasyon Bitti")
    print("Best value:", study_a3c.best_value)
    print("Best params:", study_a3c.best_params)

    # PPO optimizasyon çalışmasını başlat
    study_ppo = optuna.create_study(direction="maximize")
    study_ppo.optimize(objective_ppo, n_trials=n_trails)

    print("PPO Optimizasyon Bitti")
    print("Best value:", study_ppo.best_value)
    print("Best params:", study_ppo.best_params)
