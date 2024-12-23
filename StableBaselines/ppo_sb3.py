import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Ortamı oluştur
env = gym.make("LunarLander-v3")

# PPO modelini oluştur
model = PPO("MlpPolicy", env, verbose=1)

# Modeli eğit
print("Model eğitiliyor...")
model.learn(total_timesteps=100000)  # Toplam 100.000 adım boyunca eğitim

# Modeli değerlendirme
print("Model değerlendiriliyor...")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)
print(f"Ortalama ödül: {mean_reward} +/- {std_reward}")

# Eğitilmiş modeli test et
obs, info = env.reset()  # Ortamı sıfırla
for _ in range(1000):  # 1000 adım boyunca ortamda ajanı test et
    action, _ = model.predict(obs, deterministic=True)  # Ajanın aksiyon tahmini
    obs, reward, terminated, truncated, info = env.step(action)  # Ortama aksiyonu uygula
    env.render()  # Ortamı görselleştir
    if terminated or truncated:
        obs, info = env.reset()  # Bölüm bittiğinde ortamı sıfırla

# Ortamı kapat
env.close()
