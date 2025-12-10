import numpy as np

# Constants
BOARD_WIDTH = 7
BOARD_HEIGHT = 6

class TestConnect4Game:
    def __init__(self, player1_ai=None, player2_ai=None):
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        self.current_player = 1  # 1 for red, 2 for yellow
        self.game_over = False
        self.winner = None

        # stats
        self.player1_moves = 0
        self.player1_cols = []
        self.player2_moves = 0
        self.player2_cols = []
        
        # AI players (None means human player)
        self.player1_ai = player1_ai
        self.player2_ai = player2_ai
        
    def is_valid_move(self, col):
        """Check if a move is valid in the given column"""
        return self.board[0][col] == 0
    
    def make_move(self, col):
        """Make a move in the given column"""
        if not self.is_valid_move(col):
            return False
            
        # Find the lowest empty row in the column
        for row in range(BOARD_HEIGHT - 1, -1, -1):
            if self.board[row][col] == 0:
                self.board[row][col] = self.current_player
                # update stats
                if self.current_player == 1:
                    self.player1_moves += 1
                    self.player1_cols.append(col)
                else:
                    self.player2_moves += 1
                    self.player2_cols.append(col)
                break
        
        # Check for win or draw
        if self.check_win():
            self.game_over = True
            self.winner = self.current_player
        elif self.is_board_full():
            self.game_over = True
            self.winner = 0  # Draw
        else:
            # Switch players
            self.current_player = 2 if self.current_player == 1 else 1
        
        return True
    
    def check_win(self):
        """Check if the current player has won"""
        # Check horizontal
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH - 3):
                if (self.board[row][col] == self.current_player and
                    self.board[row][col+1] == self.current_player and
                    self.board[row][col+2] == self.current_player and
                    self.board[row][col+3] == self.current_player):
                    return True
        
        # Check vertical
        for row in range(BOARD_HEIGHT - 3):
            for col in range(BOARD_WIDTH):
                if (self.board[row][col] == self.current_player and
                    self.board[row+1][col] == self.current_player and
                    self.board[row+2][col] == self.current_player and
                    self.board[row+3][col] == self.current_player):
                    return True
        
        # Check diagonal (top left to bottom right)
        for row in range(BOARD_HEIGHT - 3):
            for col in range(BOARD_WIDTH - 3):
                if (self.board[row][col] == self.current_player and
                    self.board[row+1][col+1] == self.current_player and
                    self.board[row+2][col+2] == self.current_player and
                    self.board[row+3][col+3] == self.current_player):
                    return True
        
        # Check diagonal (top right to bottom left)
        for row in range(BOARD_HEIGHT - 3):
            for col in range(3, BOARD_WIDTH):
                if (self.board[row][col] == self.current_player and
                    self.board[row+1][col-1] == self.current_player and
                    self.board[row+2][col-2] == self.current_player and
                    self.board[row+3][col-3] == self.current_player):
                    return True
        
        return False
    
    def is_board_full(self):
        """Check if the board is full (draw condition)"""
        return np.all(self.board != 0)
    
    def reset_game(self):
        """Reset the game to initial state"""
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        # Reset stats
        self.player1_moves = 0
        self.player1_cols = []
        self.player2_moves = 0
        self.player2_cols = []
    
    def get_current_ai(self):
        """Get the AI for the current player"""
        if self.current_player == 1:
            return self.player1_ai
        else:
            return self.player2_ai
    
    def make_ai_move(self):
        """Make a move using the current player's AI"""
        ai = self.get_current_ai()
        if ai is not None:
            col = ai.get_move(self.board)
            return self.make_move(col)
        return False
    
    def run(self):
        """Main game loop (headless)"""
        move_count = 0
        max_moves = BOARD_WIDTH * BOARD_HEIGHT  # Safety limit to prevent infinite loops
        
        while not self.game_over and move_count < max_moves:
            current_ai = self.get_current_ai()
            
            if current_ai is not None:
                self.make_ai_move()
                move_count += 1
            else:
                break
            
            if self.game_over:
                break
        
        return self.winner

def main():
    """Main function to start the game"""
    game = TestConnect4Game()
    game.run()

if __name__ == "__main__":
    main()
