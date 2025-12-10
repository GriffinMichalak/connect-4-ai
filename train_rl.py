"""
Script to run training model for Reinforcement Learning AI
"""
import argparse
import os
from models.rl_cnn_ai import CNNRLAI

# Helper to format number of games
def format_games(num):
    if num >= 1_000_000:
        return f"{num // 1_000_000}M"
    if num >= 1000:
        return f"{num // 1000}k"
    return str(num)

def main():
    parser = argparse.ArgumentParser(description="Train CNN RL Agent for Connect4")
    parser.add_argument("--mode", type=str, default="random",
                        choices=["random", "self", "heuristic"],
                        help="Training opponent type")
    parser.add_argument("--games", type=int, default=None,
                        help="Number of games to train (default = model default)")
    parser.add_argument("--eps", type=float, default=None,
                        help="Epsilon decay factor (default = model default)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use: cpu / cuda / cuda:0 (default = auto)")
    args = parser.parse_args()

    rl = CNNRLAI(player_id=1, device=args.device)
    num_games = args.games if args.games is not None else 10000
    if args.eps is not None:
        rl.epsilon_decay = args.eps
    
    # For printing & saving
    mode = args.mode
    eps_decay_str = f"{rl.epsilon_decay}"

    rl.train(num_games=num_games, mode=mode)

    games_label = format_games(num_games)
    model_path = f"CNN_models/cnnrl_{mode}_{games_label}_{eps_decay_str}.pt"
    rl.save_model(model_path)

    print(f"\nModel saved to: {model_path}\n")

if __name__ == "__main__":
    main()