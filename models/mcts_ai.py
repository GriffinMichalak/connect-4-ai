"""
Monte Carlo Tree Search AI for Connect 4 Game
This module implements the MCTSAI class using Monte Carlo Tree Search algorithm
"""

from .base_ai import Connect4AI

class MCTSAI(Connect4AI):
    """Monte Carlo Tree Search AI for Connect 4"""
    
    def __init__(self, player_id):
        super().__init__(player_id)