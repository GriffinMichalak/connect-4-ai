"""
Evaluation script for trained RL models.
Runs each model against the RandomAI and prints win/loss/draw statistics.
"""
import os
import matplotlib.pyplot as plt
from connect4 import Connect4Game
from models.rl_ai import ReinforcementLearningAI
from models.simple_ais import RandomAI

def evaluate_model(model_path, num_games = 1000):
    """Evaluate a single RL model against the Random AI"""
    print(f"Evaluating model: {model_path} ({num_games} games)")

    rl = ReinforcementLearningAI(
        player_id= 1,
        training=False,
        q_table_path=model_path
    )

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

    opponent, games = parse_model_name(os.path.basename(model_path))
    graph_label = f"{opponent} ({games//1000}k)" if games else opponent

    # Print results
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Draws: {draws}")
    print(f"Win rate:: {win_rate:.2f}%")

    return (graph_label, win_rate)

def plot_results(results):
    """Bar chart comparing win rates for models"""
    model_names = [name for name, _ in results]
    win_rates = [rate for _, rate in results]

    # Chart
    plt.figure(figsize=(8, 6))
    plt.bar(model_names, win_rates, color="blue")
    plt.title("RL Model Win Rate Against Random AI")
    plt.ylabel("Win Rate (%)")
    plt.xlabel("Training Model")
    plt.tight_layout()
    plt.savefig("rl_evaluation_results.png")

def main():
    folder = "qtables"

    if not os.path.exists(folder):
        print("No qtables folder found.")
        return
    
    # Get all model files
    models = [file for file in os.listdir(folder) if file.endswith(".pkl")]

    if not models:
        print("No saved models found.")
        return
    
    # Evaluate all models at once (WILL ADD FUNCTIONALITY TO EVALUATE SINGLE MODELS LATER)
    results = []
    for model in models:
        results.append(evaluate_model(os.path.join(folder, model)))
    
    plot_results(results)
    print("Evaluation completed.")

def parse_model_name(model_name):
    """
    To parse model name based on filename.
    Expected format: qtable_<opponent>_<num_games>.pkl
    """
    name = model_name.replace(".pkl", "")
    parts = name.split("_")

    opponent = parts[1]
    
    # Parse game count
    try:
        games = int(parts[2])
    except ValueError:
        games = None

    return opponent, games

    
if __name__ == "__main__":
    main()