"""
Reinforcement Learning AI for Connect 4 Game
This module implements the ReinforcementLearningAI class using Q-learning
"""

from .base_ai import Connect4AI

class ReinforcementLearningAI(Connect4AI):
    """Reinforcement Learning AI for Connect 4"""
    
    def __init__(self, player_id):
        super().__init__(player_id)