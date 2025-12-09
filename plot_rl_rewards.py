import numpy as np
import matplotlib.pyplot as plt

def plot_training_history(path):
    data = np.load(path)
    rewards = data['episode_rewards']
    avg = data['moving_avg_rewards']

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Episode Reward", alpha=0.3)
    plt.plot(avg, label="Moving Avg (100 episodes)", linewidth=2)
    plt.title("Training Reward Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("reward_plot_10k_heuristic_0.9999.png")

if __name__ == "__main__":
    plot_training_history("training_history/training_10000_heuristic_0.9999.npz")