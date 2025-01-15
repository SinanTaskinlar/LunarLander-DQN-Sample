import torch

import Models.A3C as A3C
import Models.DQN as DQN
import Models.PPO as PPO
import Optimizer.Bayes as bayes
import Utilities.Utils as u


def main():
    print(f'PyTorch version: {torch.__version__}')
    cudnn_version = torch.backends.cudnn.version()
    print(f'CUDNN version: {cudnn_version if cudnn_version else "None"}')

    if torch.cuda.is_available():
        print(f'Device: {torch.cuda.get_device_name(0)}')
    else:
        print("CUDA desteklenmiyor, CPU kullanılacak.")

    max_episodes = 5000
    environment_name = "LunarLander-v3"
    # render_mode = "human"
    # render_mode = "rgb_array"
    render_mode = None

    print("Optimizasyon başlıyor.")
    bayes.start_optimization()

    dqn_rewards, dqn_time = DQN.DQNstart(environment_name, render_mode, max_episodes)
    print(f"DQN time: {dqn_time:.1f} seconds")

    a3c_rewards, a3c_time = A3C.A3Cstart(environment_name, render_mode, max_episodes)
    print(f"A3C time: {a3c_time:.1f} seconds")

    ppo_rewards, ppo_time = PPO.PPOstart(environment_name, render_mode, max_episodes)
    print(f"PPO time: {ppo_time:.1f} seconds")

    u.plot_all_algorithms(dqn_rewards, ppo_rewards, a3c_rewards)
    print("Tüm eğitimler tamamlandı.")
    u.print_all_times(dqn_time, a3c_time, ppo_time)

    # Kaydedilen modeli yüklemek

    # dqn_model = DQNModel(a3c_env.observation_space.shape[0], a3c_env.action_space.n).to(device)
    # load_model(dqn_model, "dqn_model_500.pth")
    # ppo_model = PPOModel(a3c_env.observation_space.shape[0], a3c_env.action_space.n).to(device)
    # load_model(ppo_model, "ppo_model_500.pth")
    # a3c_model = A3CModel(a3c_env.observation_space.shape[0], a3c_env.action_space.n).to(device)
    # load_model(a3c_model, "a3c_model.pth")


if __name__ == "__main__":
    main()
