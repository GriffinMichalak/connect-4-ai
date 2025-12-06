"""
Base AI Interface for Connect 4 Game
This module provides the abstract base class for implementing different AI strategies
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

    def get_opponent_id(self):
        """Gets the player ID of the other player"""
        return 1 if self.player_id == 2 else 2

    def check_win(self, board, player_id):
        """Check if the given player (`player_id`) has won (got 4 in a row)"""
        # check for a horizontal win
        for row in range(board.shape[0]):
            for col in range(board.shape[1] - 3):
                if (board[row][col] == player_id and
                    board[row][col+1] == player_id and
                    board[row][col+2] == player_id and
                    board[row][col+3] == player_id):
                    return True
        
        # check for a vertical win
        for row in range(board.shape[0] - 3):
            for col in range(board.shape[1]):
                if (board[row][col] == player_id and
                    board[row+1][col] == player_id and
                    board[row+2][col] == player_id and
                    board[row+3][col] == player_id):
                    return True
        
        # check for a diagonal win (top left to bottom right)
        for row in range(board.shape[0] - 3):
            for col in range(board.shape[1] - 3):
                if (board[row][col] == player_id and
                    board[row+1][col+1] == player_id and
                    board[row+2][col+2] == player_id and
                    board[row+3][col+3] == player_id):
                    return True
        
        # check for a diagonal win (top right to bottom left)
        for row in range(board.shape[0] - 3):
            for col in range(3, board.shape[1]):
                if (board[row][col] == player_id and
                    board[row+1][col-1] == player_id and
                    board[row+2][col-2] == player_id and
                    board[row+3][col-3] == player_id):
                    return True
        
        return False

    def simulate_move(self, board, move_col, player_id):
        """Returns what `board` would look like if we made the given `move`"""
        new_board = board.copy()
            
        # Find the lowest empty row in the column
        board_height = new_board.shape[0]
        for row in range(board_height - 1, -1, -1):
            if new_board[row][move_col] == 0:
                new_board[row][move_col] = player_id
                break
        
        return new_board

    def count_potential_wins(self, board, player_id):
        """Counts 3 in a rows with a space for a 4th chip"""
        count = 0
        
        # horizontal
        for row in range(board.shape[0]):
            for col in range(board.shape[1] - 3):
                sequence = [board[row][col+i] for i in range(4)]
                if sequence.count(player_id) == 3 and sequence.count(0) == 1:
                    count += 1
        
        # vertical
        for row in range(board.shape[0] - 3):
            for col in range(board.shape[1]):
                sequence = [board[row+i][col] for i in range(4)]
                if sequence.count(player_id) == 3 and sequence.count(0) == 1:
                    count += 1
        
        # diagonal (top left to bottom right)
        for row in range(board.shape[0] - 3):
            for col in range(board.shape[1] - 3):
                sequence = [board[row+i][col+i] for i in range(4)]
                if sequence.count(player_id) == 3 and sequence.count(0) == 1:
                    count += 1
        
        # diagonal (top right to bottom left)
        for row in range(board.shape[0] - 3):
            for col in range(3, board.shape[1]):
                sequence = [board[row+i][col-i] for i in range(4)]
                if sequence.count(player_id) == 3 and sequence.count(0) == 1:
                    count += 1
        
        return count

    def in_a_row(self, n, board, player_id):
        """Counts `n`-in-a-row sequences for the given player"""
        if n < 2:
            return [] # idc about 1 or 0 in a row
        
        sequences = []
        
        # horizonta
        for row in range(board.shape[0]):
            for col in range(board.shape[1] - n + 1):
                if all(board[row][col + i] == player_id for i in range(n)):
                    sequences.append([(row, col + i) for i in range(n)])
        
        # vertical
        for row in range(board.shape[0] - n + 1):
            for col in range(board.shape[1]):
                if all(board[row + i][col] == player_id for i in range(n)):
                    sequences.append([(row + i, col) for i in range(n)])
        
        # diagonal (top left to bottom right)
        for row in range(board.shape[0] - n + 1):
            for col in range(board.shape[1] - n + 1):
                if all(board[row + i][col + i] == player_id for i in range(n)):
                    sequences.append([(row + i, col + i) for i in range(n)])
        
        # diagonal (top right to bottom left)
        for row in range(board.shape[0] - n + 1):
            for col in range(n - 1, board.shape[1]):
                if all(board[row + i][col - i] == player_id for i in range(n)):
                    sequences.append([(row + i, col - i) for i in range(n)])
        
        return sequences
