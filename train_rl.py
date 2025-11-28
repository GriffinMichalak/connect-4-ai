"""
Simple script to run training model for Reinforcement Learning AI
"""
from models.rl_ai import ReinforcementLearningAI

def main():
    ai = ReinforcementLearningAI(
        player_id=1,
        training=True
    )

    ai.train(num_games=1000000, mode="random")

if __name__ == "__main__":
    main()