"""
Minimax AI with Alpha-Beta Pruning for Connect 4 Game
This module implements the MinimaxAI class with alpha-beta pruning optimization
"""

from .base_ai import Connect4AI

class MinimaxAI(Connect4AI):
    """Minimax AI with Alpha-Beta pruning for Connect 4"""
    
    def __init__(self, player_id):
        super().__init__(player_id)