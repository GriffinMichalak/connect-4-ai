"""
Heuristic-based AI for Connect 4 Game
This module implements the HeuristicAI class using simple heuristic evaluation
"""

import numpy as np
from .base_ai import Connect4AI

class HeuristicAI(Connect4AI):
    """Simple Heuristic-based AI for Connect 4"""

    def get_move(self, board):
        valid_moves = self.get_valid_moves(board)
        
        if not valid_moves:
            return 0

        best_move = valid_moves[0]
        best_score = -1

        for move in valid_moves:
            # 1. simulate placing the piece & get what-if score
            temp_board = self.simulate_move(board, move, self.player_id)
            score = self.heuristic_score(temp_board)
            # 2. set current best move
            if score > best_score:
                best_move = move
                best_score = score

        # 4. return the move with the highest score
        return best_move

    def heuristic_score(self, board):
        """generates a score the given `board` based on Connect4 heuristics"""
        score = 0
        
        # 1. check if you won
        if self.check_win(board, self.player_id):
            return 10000
        # 2. check if your opponent won
        if self.check_win(board, self.get_opponent_id()):
            return -10000  # Loss
        
        # 3. count 3 in a rows. the more, teh better
        my_sequences = self.in_a_row(3, board, self.player_id)
        opp_sequences = self.in_a_row(3, board, self.get_opponent_id())
        
        score += len(my_sequences) * 1000
        score -= len(opp_sequences) * 1000
        
        # 4. count 2 in a rows. the more, the better
        my_2_sequences = self.in_a_row(2, board, self.player_id)
        opp_2_sequences = self.in_a_row(2, board, self.get_opponent_id())
        
        score += len(my_2_sequences) * 100
        score -= len(opp_2_sequences) * 100
        
        # 5. the closer the placement is to the center, the better
        center_cols = [2, 3, 4]
        for row in range(board.shape[0]):
            for col in center_cols:
                if board[row][col] == self.player_id:
                    score += 10
                elif board[row][col] == self.get_opponent_id():
                    score -= 10
        
        # Look for potential 4-in-a-row opportunities
        score += self.count_potential_wins(board, self.player_id) * 50
        score -= self.count_potential_wins(board, self.get_opponent_id()) * 50
        
        return score