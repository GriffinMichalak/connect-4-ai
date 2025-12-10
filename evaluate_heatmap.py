"""
Generate a win-rate heatmap by having all major Connect4 AIs
play against each other.

Models included:
- MCTS
- Minimax
- Minimax Alpha-Beta
- Heuristic
- CNN-RL (self-play 500k)
- CNN-RL (heuristic 500k)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from connect4 import Connect4Game
from models import HeuristicAI, MinimaxAI, MinimaxABAI, MCTSAI
from models.rl_cnn_ai import CNNRLAI

def play_matchup(ai1, ai2, num_games=10):
    """
    Play between two AIs.
    ai1 = Player 1
    ai2 = Player 2
    Returns win rate of ai1 (as a float 0–1).
    """
    wins = 0
    draws = 0

    for _ in range(num_games):
        game = Connect4Game(
            player1_ai=ai1,
            player2_ai=ai2,
            screen_off=True,
            sim=True
        )
        winner = game.run()

        if winner == 1:
            wins += 1
        elif winner == 0:
            draws += 1

    return wins / num_games

def load_all_agents():
    """
    Load all AI models
    """
    agents = {}

    agents["MCTS"] = MCTSAI(player_id=1)
    agents["Minimax"] = MinimaxAI(player_id=1)
    agents["MinimaxAB"] = MinimaxABAI(player_id=1)
    agents["Heuristic"] = HeuristicAI(player_id=1)

    # CNN models (update paths as needed)
    cnn_self_path = "CNN_models/cnnrl_self_500k_0.999995.pt"
    cnn_heur_path = "CNN_models/cnnrl_heuristic_500k_0.999995.pt"

    cnn_self = CNNRLAI(player_id=1)
    cnn_self.load_model(cnn_self_path)
    cnn_self.epsilon = 0.0

    cnn_heur = CNNRLAI(player_id=1)
    cnn_heur.load_model(cnn_heur_path)
    cnn_heur.epsilon = 0.0

    agents["CNN Self 500k"] = cnn_self
    agents["CNN Heuristic 500k"] = cnn_heur

    return agents

def generate_heatmap(num_games_per_match=5):
    """
    To run all matchups and generate the heatmap
    """
    agents = load_all_agents()
    names = list(agents.keys())
    n = len(names)

    win_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                win_matrix[i][j] = 0.50  # Tie with itself
                continue

            print(f"\n=== {names[i]} (P1) vs {names[j]} (P2) ===")
            # Re-initialize agents with correct player IDs
            ai1 = type(agents[names[i]])(player_id=1)
            ai2 = type(agents[names[j]])(player_id=2)

            # Reload CNN weights if needed
            if "CNN Self" in names[i]:
                ai1.load_model("CNN_models/cnnrl_self_500k_0.999995.pt")
                ai1.epsilon = 0.0
            if "CNN Heuristic" in names[i]:
                ai1.load_model("CNN_models/cnnrl_heuristic_500k_0.999995.pt")
                ai1.epsilon = 0.0
            if "CNN Self" in names[j]:
                ai2.load_model("CNN_models/cnnrl_self_500k_0.999995.pt")
                ai2.epsilon = 0.0
            if "CNN Heuristic" in names[j]:
                ai2.load_model("CNN_models/cnnrl_heuristic_500k_0.999995.pt")
                ai2.epsilon = 0.0

            win_rate = play_matchup(ai1, ai2, num_games=num_games_per_match)
            win_matrix[i][j] = win_rate

    # Create heatmap (win rates shown as percentages)
    plt.figure(figsize=(10, 8))

    # Convert win rates 0–1 to string "40%"
    percent_labels = np.vectorize(lambda x: f"{int(round(x * 100))}%")(win_matrix)

    # Fill diagonal valeus (self vs self) with mid-value and N/A label
    for i in range(len(win_matrix)):
        win_matrix[i][i] = 0.5  # yellow midpoint
        percent_labels[i][i] = "N/A"

    sns.heatmap(
        win_matrix,
        annot=percent_labels,
        fmt="",
        xticklabels=names,
        yticklabels=names,
        cmap="RdYlGn", # Green = win, red = loss
        vmin=0,
        vmax=1,
        cbar=True,
        linewidths=0.5,
        linecolor="white",
    )

    plt.title(f"AI vs AI Win Rate Heatmap ({num_games_per_match} games per matchup)\n"
              f"Cell = Win rate of Row AI vs Column AI",)
    plt.tight_layout()
    plt.savefig("ai_vs_ai_heatmap.png")
    print("\nSaved heatmap to ai_vs_ai_heatmap.png\n")

if __name__ == "__main__":
    generate_heatmap(num_games_per_match=100)
