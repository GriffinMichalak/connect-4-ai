"""
Evaluation script for trained RL models.
Runs each model against the RandomAI and prints win/loss/draw statistics.
"""
import os
import matplotlib.pyplot as plt
from connect4 import Connect4Game
from models.rl_ai import ReinforcementLearningAI
from models.rl_cnn_ai import CNNRLAI
from models.simple_ais import RandomAI

def evaluate_model(model_path, num_games = 1000):
    """Evaluate a single RL model against the Random AI"""
    print(f"Evaluating model: {model_path} ({num_games} games)")

    rl = CNNRLAI(player_id=1)
    rl.load_model(model_path) # load .pt weight file
    rl.epsilon = 0.0 # Disable exploration during evaluation

    random_ai = RandomAI(player_id=2)

    wins = 0
    losses = 0
    draws = 0

    # Simulate games
    for _ in range (num_games):
        game = Connect4Game(
            player1_ai=rl,
            player2_ai=random_ai,
            screen_off=True,
            sim=True
        )
        winner = game.run()
        if winner == 1:
            wins += 1
        elif winner == 2:
            losses += 1
        else:
            draws += 1

    # Winrate
    win_rate = wins / num_games * 100

    label = parse_model_name(os.path.basename(model_path))

    # Print results
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Draws: {draws}")
    print(f"Win rate: {win_rate:.2f}%")

    return (label, win_rate)

def plot_results(results):
    """Bar chart comparing win rates for models"""

    def extract_games(label):
            try:
                inside = label.split("(")[1].split(",")[0]  # "50k"
                if inside.endswith("k"):
                    return int(inside[:-1]) * 1000
                return int(inside)
            except:
                return 0
    
    results = sorted(results, key=lambda x: extract_games(x[0]))
            
    model_names = [name for name, _ in results]
    win_rates = [rate for _, rate in results]

    # Chart
    plt.figure(figsize=(8, 6))
    plt.bar(model_names, win_rates, color="blue")
    plt.title("RL Model Win Rate Against Random AI")
    plt.ylabel("Win Rate (%)")
    plt.xlabel("Training Model")
    plt.tight_layout()
    plt.savefig("rl_evaluation_results1.png")

def parse_model_name(filename):
    """
    To parse model name based on filename.
    Expected format: cnnrl_[opponent]_[games]_[epsilon_decay].pt
    """
    base = filename.replace(".pt", "")
    parts = base.split("_")
    
    if len(parts) == 4 and parts[0] == "cnn":
        opponent = parts[1]
        try:
            games = int(parts[2])
            games_label = f"{games // 1000}k"
        except:
            games_label = parts[2]
        epsilon_decay = parts[3]
        return f"{opponent} ({games_label}, eps={epsilon_decay})"

    return base

def main():
    folder = "CNN_models"

    if not os.path.exists(folder):
        print("No CNN_models folder found.")
        return
    
    # Get all model files
    models = [file for file in os.listdir(folder) if file.endswith(".pt")]
    if not models:
        print("No CNN .pt models found.")
        return
    
    # Evaluate all models at once
    results = []
    for model in models:
        results.append(evaluate_model(os.path.join(folder, model)))
    
    plot_results(results)
    print("Evaluation completed.")
    
if __name__ == "__main__":
    main()