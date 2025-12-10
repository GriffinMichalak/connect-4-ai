"""
Monte Carlo Tree Search AI for Connect 4 Game
This module implements the MCTSAI class using Monte Carlo Tree Search algorithm
"""

import numpy as np
import math
from .base_ai import Connect4AI

class MCTSNode:
    """Represents a node in the MCTS search tree"""
    
    def __init__(self, board, move=None, parent=None):
        self.board = board.copy()
        self.move = move  # The move that led to this state
        self.parent = parent
        
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.untried_moves = self._get_valid_moves()
        
    def _get_valid_moves(self):
        """Get all valid column moves for current board"""
        return [col for col in range(7) if self.board[0][col] == 0]
    
    def _get_current_player(self):
        """Determine whose turn it is based on piece count"""
        count1 = np.sum(self.board == 1)
        count2 = np.sum(self.board == 2)
        # If equal pieces, player 1's turn; if player 1 has more, player 2's turn
        return 1 if count1 == count2 else 2
    
    def is_fully_expanded(self):
        """Check if all possible moves have been tried"""
        return len(self.untried_moves) == 0
    
    def is_terminal(self):
        """Check if this is a terminal node (game over)"""
        # Check if either player has won
        if self._check_win(1) or self._check_win(2):
            return True
        # Check if board is full
        return len(self._get_valid_moves()) == 0
    
    def _check_win(self, player_id):
        """Check if the given player has won"""
        board = self.board
        
        # Horizontal
        for row in range(6):
            for col in range(4):
                if (board[row][col] == player_id and
                    board[row][col+1] == player_id and
                    board[row][col+2] == player_id and
                    board[row][col+3] == player_id):
                    return True
        
        # Vertical
        for row in range(3):
            for col in range(7):
                if (board[row][col] == player_id and
                    board[row+1][col] == player_id and
                    board[row+2][col] == player_id and
                    board[row+3][col] == player_id):
                    return True
        
        # Diagonal (top-left to bottom-right)
        for row in range(3):
            for col in range(4):
                if (board[row][col] == player_id and
                    board[row+1][col+1] == player_id and
                    board[row+2][col+2] == player_id and
                    board[row+3][col+3] == player_id):
                    return True
        
        # Diagonal (top-right to bottom-left)
        for row in range(3):
            for col in range(3, 7):
                if (board[row][col] == player_id and
                    board[row+1][col-1] == player_id and
                    board[row+2][col-2] == player_id and
                    board[row+3][col-3] == player_id):
                    return True
        
        return False
    
    def best_child(self, c_param=1.414):
        """Select best child using UCB1 formula"""
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                # Prioritize unvisited children
                choices_weights.append(float('inf'))
            else:
                exploit = child.wins / child.visits
                explore = c_param * math.sqrt(math.log(self.visits) / child.visits)
                choices_weights.append(exploit + explore)
        
        return self.children[np.argmax(choices_weights)]
    
    def expand(self):
        """Expand tree by creating a new child node"""
        move = self.untried_moves.pop()
        new_board = self._apply_move(move)
        child_node = MCTSNode(new_board, move=move, parent=self)
        self.children.append(child_node)
        return child_node
    
    def _apply_move(self, col):
        """Apply a move to the board and return new board state"""
        new_board = self.board.copy()
        current_player = self._get_current_player()
        
        for row in range(5, -1, -1):
            if new_board[row][col] == 0:
                new_board[row][col] = current_player
                break
        
        return new_board
    
    def rollout(self, ai_player_id):
        """Simulate a random game from this node to completion"""
        current_board = self.board.copy()
        current_player = self._get_current_player()
        
        # Play random moves until game ends
        for _ in range(100):  # Max 100 moves to prevent infinite loops
            # Check for terminal state
            if self._check_win_on_board(current_board, 1):
                return 1 if ai_player_id == 1 else -1
            if self._check_win_on_board(current_board, 2):
                return 1 if ai_player_id == 2 else -1
            
            # Get valid moves
            valid_moves = [col for col in range(7) if current_board[0][col] == 0]
            if not valid_moves:
                return 0  # Draw
            
            # Make random move
            move = np.random.choice(valid_moves)
            for row in range(5, -1, -1):
                if current_board[row][move] == 0:
                    current_board[row][move] = current_player
                    break
            
            # Switch player
            current_player = 2 if current_player == 1 else 1
        
        return 0  # Draw if we hit the move limit
    
    def _check_win_on_board(self, board, player_id):
        """Check if player has won on given board"""
        # Horizontal
        for row in range(6):
            for col in range(4):
                if (board[row][col] == player_id and
                    board[row][col+1] == player_id and
                    board[row][col+2] == player_id and
                    board[row][col+3] == player_id):
                    return True
        
        # Vertical
        for row in range(3):
            for col in range(7):
                if (board[row][col] == player_id and
                    board[row+1][col] == player_id and
                    board[row+2][col] == player_id and
                    board[row+3][col] == player_id):
                    return True
        
        # Diagonal (top-left to bottom-right)
        for row in range(3):
            for col in range(4):
                if (board[row][col] == player_id and
                    board[row+1][col+1] == player_id and
                    board[row+2][col+2] == player_id and
                    board[row+3][col+3] == player_id):
                    return True
        
        # Diagonal (top-right to bottom-left)
        for row in range(3):
            for col in range(3, 7):
                if (board[row][col] == player_id and
                    board[row+1][col-1] == player_id and
                    board[row+2][col-2] == player_id and
                    board[row+3][col-3] == player_id):
                    return True
        
        return False
    
    def backpropagate(self, result):
        """Update node statistics based on simulation result
        result: 1 for AI win, -1 for opponent win, 0 for draw
        """
        self.visits += 1
        
        # Convert result to wins (result is from AI's perspective)
        if result == 1:
            self.wins += 1.0
        elif result == 0:
            self.wins += 0.5
        # result == -1 means loss, add 0
        
        if self.parent:
            # Flip the result when propagating up (opponent's perspective)
            self.parent.backpropagate(-result)


class MCTSAI(Connect4AI):
    """Monte Carlo Tree Search AI for Connect 4"""
    
    def __init__(self, player_id, num_simulations=2000, c_param=1.414):
        super().__init__(player_id)
        self.num_simulations = num_simulations
        self.c_param = c_param
    
    def get_move(self, board):
        """Get the best move using MCTS"""
        valid_moves = self.get_valid_moves(board)
        
        if not valid_moves:
            return 0
        
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # Create root node
        root = MCTSNode(board)
        
        # Run MCTS simulations
        for _ in range(self.num_simulations):
            node = self._select(root)
            
            # Expand if not terminal and not fully expanded
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
            
            # Rollout from this node
            result = node.rollout(self.player_id)
            
            # Backpropagate
            node.backpropagate(result)
        
        # Choose the move with best win rate
        best_child = None
        best_value = -float('inf')
        for child in root.children:
            if child.visits > 0:
                win_rate = child.wins / child.visits
                if win_rate > best_value:
                    best_value = win_rate
                    best_child = child
        
        return best_child.move if best_child else valid_moves[0]
    
    def _select(self, node):
        """Selection phase: traverse tree using UCB1"""
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node
            else:
                node = node.best_child(c_param=self.c_param)
        return node
