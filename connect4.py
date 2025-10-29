import pygame
import sys
import numpy as np
from models import Connect4AI, RandomAI, HumanPlayer, MinimaxAI, MCTSAI, ReinforcementLearningAI, HeuristicAI

pygame.init()

# Constants
BOARD_WIDTH = 7
BOARD_HEIGHT = 6
CELL_SIZE = 80
WINDOW_WIDTH = BOARD_WIDTH * CELL_SIZE
WINDOW_HEIGHT = (BOARD_HEIGHT + 1) * CELL_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
LIGHT_BLUE = (173, 216, 230)

class Connect4Game:
    def __init__(self, player1_ai=None, player2_ai=None):
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        self.current_player = 1  # 1 for red, 2 for yellow
        self.game_over = False
        self.winner = None
        
        # AI players (None means human player)
        self.player1_ai = player1_ai
        self.player2_ai = player2_ai
        
        # Initialize display
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Connect 4")
        self.clock = pygame.time.Clock()
        
        # Font for text
        self.font = pygame.font.Font(None, 36)
        
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
        
        # Check diagonal (top-left to bottom-right)
        for row in range(BOARD_HEIGHT - 3):
            for col in range(BOARD_WIDTH - 3):
                if (self.board[row][col] == self.current_player and
                    self.board[row+1][col+1] == self.current_player and
                    self.board[row+2][col+2] == self.current_player and
                    self.board[row+3][col+3] == self.current_player):
                    return True
        
        # Check diagonal (top-right to bottom-left)
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
    
    def draw_board(self):
        """Draw the game board"""
        self.screen.fill(LIGHT_BLUE)
        
        # Draw the board background
        pygame.draw.rect(self.screen, BLUE, (0, CELL_SIZE, WINDOW_WIDTH, WINDOW_HEIGHT - CELL_SIZE))
        
        # Draw the grid
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                x = col * CELL_SIZE
                y = (row + 1) * CELL_SIZE
                pygame.draw.circle(self.screen, WHITE, 
                                 (x + CELL_SIZE // 2, y + CELL_SIZE // 2), 
                                 CELL_SIZE // 2 - 5)
        
        # Draw the pieces
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                if self.board[row][col] != 0:
                    x = col * CELL_SIZE + CELL_SIZE // 2
                    y = (row + 1) * CELL_SIZE + CELL_SIZE // 2
                    color = RED if self.board[row][col] == 1 else YELLOW
                    pygame.draw.circle(self.screen, color, (x, y), CELL_SIZE // 2 - 5)
        
        # Draw column indicators
        for col in range(BOARD_WIDTH):
            x = col * CELL_SIZE + CELL_SIZE // 2
            y = CELL_SIZE // 2
            if not self.game_over and self.is_valid_move(col):
                # Highlight valid columns
                pygame.draw.circle(self.screen, GRAY, (x, y), CELL_SIZE // 2 - 5)
            else:
                pygame.draw.circle(self.screen, WHITE, (x, y), CELL_SIZE // 2 - 5)
    
    def draw_ui(self):
        """Draw the user interface elements"""
        if not self.game_over:
            # Show current player
            current_ai = self.get_current_ai()
            if current_ai is not None:
                player_text = f"AI Player {self.current_player}'s Turn"
            else:
                player_text = f"Player {self.current_player}'s Turn"
            
            player_color = RED if self.current_player == 1 else YELLOW
            text_surface = self.font.render(player_text, True, player_color)
            text_rect = text_surface.get_rect(center=(WINDOW_WIDTH // 2, CELL_SIZE // 2))
            self.screen.blit(text_surface, text_rect)
        else:
            # Show game over message
            if self.winner == 0:
                game_text = "It's a Draw!"
                text_color = BLACK
            else:
                winner_ai = self.player1_ai if self.winner == 1 else self.player2_ai
                if winner_ai is not None:
                    game_text = f"AI Player {self.winner} Wins!"
                else:
                    game_text = f"Player {self.winner} Wins!"
                text_color = RED if self.winner == 1 else YELLOW
            
            text_surface = self.font.render(game_text, True, text_color)
            text_rect = text_surface.get_rect(center=(WINDOW_WIDTH // 2, CELL_SIZE // 2))
            self.screen.blit(text_surface, text_rect)
            
            # Show restart instruction
            restart_text = "Press R to restart or ESC to quit"
            restart_surface = self.font.render(restart_text, True, BLACK)
            restart_rect = restart_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 20))
            self.screen.blit(restart_surface, restart_rect)
    
    def handle_click(self, pos):
        """Handle mouse click events"""
        if self.game_over:
            return
        
        # Only allow human moves if current player is human
        current_ai = self.get_current_ai()
        if current_ai is not None:
            return
        
        x, y = pos
        col = x // CELL_SIZE
        
        if 0 <= col < BOARD_WIDTH:
            self.make_move(col)
    
    def run(self):
        """Main game loop"""
        running = True
        ai_move_timer = 0
        ai_move_delay = 1000  # 1 second delay for AI moves
        
        while running:
            current_time = pygame.time.get_ticks()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos)
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r and self.game_over:
                        self.reset_game()
                        ai_move_timer = current_time  # Reset AI timer
                    elif event.key == pygame.K_ESCAPE:
                        running = False
            
            # Handle AI moves
            if not self.game_over:
                current_ai = self.get_current_ai()
                if current_ai is not None and current_time - ai_move_timer > ai_move_delay:
                    self.make_ai_move()
                    ai_move_timer = current_time
            
            # Draw everything
            self.draw_board()
            self.draw_ui()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

def main():
    """Main function to start the game"""
    game = Connect4Game()
    game.run()

if __name__ == "__main__":
    main()
