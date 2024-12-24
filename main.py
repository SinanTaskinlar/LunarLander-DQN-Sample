import Models.PPO as PPO,Models.DQN as DQN,Models.A3C as A3C
import Utilities.Utils as Utils
import torch

def main():
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDNN version: {torch.backends.cudnn.version()}')
    print(f'Device: {torch.cuda.get_device_name()}')

    max_episodes = 5000
    environment_name = "LunarLander-v3"
    # render_mode = "human"
    # render_mode = "rgb_array"
    render_mode = None

    dqn_rewards, dqn_time = DQN.DQNstart(environment_name, render_mode, max_episodes)
    print(f"DQN time: {dqn_time}")

    a3c_rewards, a3c_time = A3C.A3Cstart(environment_name, render_mode, max_episodes)
    print(f"A3C time: {a3c_time}")

    ppo_rewards, ppo_time = PPO.PPOstart(environment_name, render_mode, max_episodes)
    print(f"PPO time: {ppo_time:.1f} seconds")

    Utils.plot_all_algorithms(dqn_rewards, ppo_rewards, a3c_rewards)

    # Kaydedilen modeli yüklemek

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # dqn_model = DQNModel(a3c_env.observation_space.shape[0], a3c_env.action_space.n).to(device)
    # load_model(dqn_model, "dqn_model_500.pth")
    # ppo_model = PPOModel(a3c_env.observation_space.shape[0], a3c_env.action_space.n).to(device)
    # load_model(ppo_model, "ppo_model_500.pth")
    # a3c_model = A3CModel(a3c_env.observation_space.shape[0], a3c_env.action_space.n).to(device)
    # load_model(a3c_model, "a3c_model.pth")

if __name__ == "__main__":
    main()