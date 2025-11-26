"""
Minimax AI without Alpha-Beta Pruning for Connect 4 Game
This module implements the MinimaxAI class without alpha-beta pruning optimization
"""

from .base_ai import Connect4AI
from .heuristic_ai import HeuristicAI

class MinimaxAI(Connect4AI):
    """Minimax AI without Alpha-Beta pruning for Connect 4"""
    
    def __init__(self, player_id, depth=4):
        super().__init__(player_id)
        self.depth = depth
        self.opponent_id = 2 if player_id == 1 else 1
        
        self.heuristic_evaluator = HeuristicAI(player_id)

    def get_move(self, board):
        """Get the best move using minimax algorithm"""
        valid_columns = self.get_valid_columns(board)
        
        if not valid_columns:
            return 0  # Should not happen in valid game state
        
        best_score = float('-inf')
        best_column = valid_columns[0]
        
        for col in valid_columns:
            # Simulate the move
            new_board = self.simulate_move_for_player(board, col, self.player_id)
            
            # Get the score for this move using minimax
            score = self.minimax(new_board, self.depth - 1, False)
            
            if score > best_score:
                best_score = score
                best_column = col
        
        return best_column
    
    def minimax(self, board, depth, is_maximizing):
        """Minimax algorithm with depth limiting"""
        # Base case: check if game is terminal or depth limit reached
        if depth == 0 or self.is_terminal_node(board):
            return self.heuristic_evaluator.heuristic_score(board)
        
        valid_columns = self.get_valid_columns(board)
        
        if is_maximizing:
            # Maximizing player (AI)
            max_score = float('-inf')
            for col in valid_columns:
                new_board = self.simulate_move_for_player(board, col, self.player_id)
                score = self.minimax(new_board, depth - 1, False)
                max_score = max(max_score, score)
            return max_score
        else:
            # Minimizing player (opponent)
            min_score = float('inf')
            for col in valid_columns:
                new_board = self.simulate_move_for_player(board, col, self.opponent_id)
                score = self.minimax(new_board, depth - 1, True)
                min_score = min(min_score, score)
            return min_score
    
    def simulate_move_for_player(self, board, col, player):
        """Simulate a move for any player and return the new board state"""
        new_board = board.copy()
        
        # Find the lowest empty row in the column
        for row in range(5, -1, -1):  # Start from bottom row
            if new_board[row][col] == 0:
                new_board[row][col] = player
                break
        
        return new_board
    
    def get_valid_columns(self, board):
        """Get all valid columns for moves"""
        return [col for col in range(7) if board[0][col] == 0]
    
    def is_terminal_node(self, board):
        """Check if the game is in a terminal state"""
        return (self.check_win(board, self.player_id) or 
                self.check_win(board, self.opponent_id) or 
                len(self.get_valid_columns(board)) == 0)
