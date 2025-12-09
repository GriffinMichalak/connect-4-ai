"""
Demo script for Connect 4 game with different AI configurations
"""

import os
from connect4 import Connect4Game
from models import HeuristicAI, MinimaxAI, RandomAI, ReinforcementLearningAI, mcts_ai, rl_cnn_ai
from models.rl_cnn_ai import CNNRLAI

def human_vs_human():
    """Two human players"""
    print("Starting Human vs Human game...")
    game = Connect4Game()
    game.run()

def human_vs_ai():
    """Human vs Random AI"""
    print("=======================================")
    print("We have several AI models. Please select one (1-5): ")
    print("=======================================")
    print("1. Minimax w/ AlphaBeta Pruning")
    print("2. Monte Carlo Tree Search")
    print("3. Reinforcement Learning")
    print("4. Simple Heuristic Search")
    print("5. Exit")
    choice = input("\nYour choice (1-5): ").strip()

    ai_player = None
    
    if choice == "1":
        # ai_player = MinimaxAI(player_id=2)
        print("Not yet implemented")
        exit()
    elif choice == "2":
        # ai_player = mcts_ai(player_id=2)
        print("Not yet implemented")
        exit()
    elif choice == "3":
        print("\nYou selected Reinforcement Learning model.")
        filename = choose_cnn_model()

        if filename is None:
            print("\nReturning to menu\n")
            return
        
        print(f"\nLoading model: {filename}")
        ai_player = CNNRLAI(player_id=2)
        ai_player.load_model(filename)
    elif choice == "4":
        ai_player = HeuristicAI(player_id=2)
    elif choice == "5":
        print("Goodbye!")
    else:
        print("Invalid choice. Please select 1-5.")
    
    print("Starting Human vs AI game...")
    game = Connect4Game(player2_ai=ai_player)
    game.run()

def ai_vs_ai():
    """Two AI players"""
    print("Starting AI vs AI game...")
    ai_player1 = RandomAI(player_id=1)
    ai_player2 = RandomAI(player_id=2)
    game = Connect4Game(player1_ai=ai_player1, player2_ai=ai_player2)
    game.run()

def choose_cnn_model():
    """Select a CNN RL model from CNN_models folder"""
    folder = "CNN_models"

    files = [f for f in os.listdir(folder) if f.endswith(".pt")]

    if not files:
        print("\nNo CNN RL model files found.")
        print("Train a CNN model and save with .pt extension.")
        return None
    
    print("\nAvailable CNN RL Models:")
    print("===================================")
    for i, file in enumerate(files, start=1):
        print(f"{i}. {file}")

    while True:
        choice = input("\nSelect a model number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return os.path.join(folder, files[int(choice) - 1])
        print("Invalid choice.")

def main():
    """Main demo function"""
    print("=======================================")
    print("Connect 4 Demo")
    print("=======================================")
    print("1. Human vs Human")
    print("2. Human vs AI")
    print("3. AI vs AI")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nSelect game mode (1-4): ").strip()
            
            if choice == "1":
                human_vs_human()
            elif choice == "2":
                human_vs_ai()
            elif choice == "3":
                ai_vs_ai()
            elif choice == "4":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please select 1-4.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
