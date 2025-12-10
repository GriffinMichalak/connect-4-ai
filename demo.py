"""
Demo script for Connect 4 game with different AI configurations
"""

import os
from connect4 import Connect4Game
from models import HeuristicAI, MinimaxAI, MinimaxABAI, RandomAI, MCTSAI
from models.rl_cnn_ai import CNNRLAI

def human_vs_human():
    """Two human players"""
    print("Starting Human vs Human game...")
    game = Connect4Game()
    game.run()

def human_vs_ai():
    """Human vs AI"""
    print("Select 0-6: ")
    print("0. Random Selection")
    print("1. Minimax (Basic)")
    print("2. Minimax w/ Alpha-Beta Pruning")
    print("3. Reinforcement Learning")
    print("4. Simple Heuristic Search")
    print("5. Monte Carlo Tree Search")
    print("6. Exit")
    choice = input("\nYour choice (1-6): ").strip()

    ai_player = None
    
    if choice == "0":
        ai_player = RandomAI(player_id=2)
        print("Selected: Random Selection 'AI'")
    elif choice == "1":
        ai_player = MinimaxAI(player_id=2)
        print("Selected: Basic Minimax AI")
    elif choice == "2":
        ai_player = MinimaxABAI(player_id=2)
        print("Selected: Minimax AI with Alpha-Beta Pruning")
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
        ai_player = MCTSAI(player_id=2)
    elif choice == "6":
        print("Goodbye!")
        return
    else:
        print("Invalid choice. Please select 1-6.")
        return
    
    print("Starting Human vs AI game...")
    game = Connect4Game(player2_ai=ai_player)
    game.run()

def ai_vs_ai():
    """Two AI players"""
    print("Starting AI vs AI game...")
    ai_player1 = MinimaxAI(player_id=1)
    ai_player2 = MinimaxAI(player_id=2)
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
    print("Connect 4 Demo")
    print("1. Human vs Human")
    print("2. Human vs AI")
    print("3. AI vs AI")
    print("4. Exit")

    running = True
    
    while running:
        try:
            choice = input("\nSelect game mode (1-4): ").strip()
            
            if choice == "1":
                human_vs_human()
                running = False
            elif choice == "2":
                human_vs_ai()
                running = False
            elif choice == "3":
                ai_vs_ai()
                running = False
            elif choice == "4":
                print("Goodbye!")
                running = False
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
