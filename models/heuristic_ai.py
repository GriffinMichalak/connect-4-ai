"""
Heuristic-based AI for Connect 4 Game
This module implements the HeuristicAI class using simple heuristic evaluation
"""

from .base_ai import Connect4AI

class HeuristicAI(Connect4AI):
    """Simple Heuristic-based AI for Connect 4"""
    
    def __init__(self, player_id):
        super().__init__(player_id)