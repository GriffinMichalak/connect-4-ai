"""
Simple AI implementations for Connect 4 Game
This module provides basic AI players like RandomAI and HumanPlayer
"""

import numpy as np
from .base_ai import Connect4AI

class RandomAI(Connect4AI):
    """Simple random AI that makes random valid moves"""
    
    def get_move(self, board):
        valid_moves = self.get_valid_moves(board)
        if not valid_moves:
            return 0
        return np.random.choice(valid_moves)

class HumanPlayer(Connect4AI):
    """Human player interface (for testing)"""
    
    def get_move(self, board):
        return 0
