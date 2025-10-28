"""
AI Interface for Connect 4 Game
This module provides a base class for implementing different AI strategies
"""

import numpy as np
from abc import ABC, abstractmethod

class Connect4AI(ABC):
    """Abstract base class for Connect 4 AI players"""
    
    def __init__(self, player_id):
        self.player_id = player_id  # 1 or 2
    
    @abstractmethod
    def get_move(self, board):
        """
        Get the AI's move for the given board state
        
        Args:
            board: 2D numpy array representing the game board
                  (0 = empty, 1 = player 1, 2 = player 2)
        
        Returns:
            int: Column index (0-6) for the move
        """
        pass
    
    def is_valid_move(self, board, col):
        """Check if a move is valid in the given column"""
        return board[0][col] == 0
    
    def get_valid_moves(self, board):
        """Get all valid moves for the current board state"""
        return [col for col in range(7) if self.is_valid_move(board, col)]

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
        # This would be handled by the GUI in practice
        return 0  # Placeholder
