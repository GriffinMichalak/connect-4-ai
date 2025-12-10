"""
NOTE: THIS MODEL IS NO LONGER IN USE. THIS MODEL WAS USED BEFORE SWITCHING TO
CNN REINFORCEMENT LEARNING. FILE KEPT TO SHOW CONTRIBUTIONS, EDITS, AND PROGESS.

Reinforcement Learning AI for Connect 4.

This class:
- Can be used in play mode (no training, only chooses moves via Q-table).
- Can be used in training mode (running many self-contained games).
- Supports different training opponents (random, heuristic, minimax, mcts).
- Stores its knowledge in a tabular Q-table (Python dict).
"""
import random
import pickle
from typing import Dict, Tuple, List, Optional
import numpy as np
from .base_ai import Connect4AI

class ReinforcementLearningAI(Connect4AI):
    """
    Q-learning based AI for Connect 4

    - If training=False -> used for real gameplay (picks best known move)
    - If training=True -> runs Q-learning games vs random
    """
    
    def __init__(
            self,
            player_id: int,
            training: bool = False,
            alpha: float = 0.3, 
            gamma: float = 0.98,
            epsilon: float = 1.0,
            epsilon_decay: float = 0.99995,
            epsilon_min: float = 0.05,
            q_table_path: Optional[str] = None
        ):
        super().__init__(player_id)

        # Board dimensions
        self.rows = 6
        self.cols = 7

        # Visitation counter for state based visitation decay
        self.visits: Dict[Tuple[int, ...], Dict[int, int]] = {}

        # Q-table: state (flattened tuple) -> {action: Q-value}
        self.q_table: Dict[Tuple[int, ...], Dict[int, float]] = {}

        self.training = training

        # Reinforcement Learning constants
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Load Q-table is path is given
        if q_table_path is not None:
            self.load_q_table(q_table_path)

    def get_move(self, board):
        """
        When playing in demo mode:
        - No exploration (epsilon ignored)
        - Always pick best known move
        - If unseen state -> random valid column
        """
        valid_moves = self.get_valid_moves(board)
        state = self.encode_state(board)

        # If model hasn't seen state -> random
        if state not in self.q_table:
            return random.choice(valid_moves)
        
        # Pick action with highest Q-value and break ties randomly
        q_values = self.q_table[state]
        best_move = max(valid_moves, key=lambda a: q_values.get(a, 0.0))
        return best_move
    
    def train(self, num_games: int = 100000, mode: str = "random"):
        assert mode in {"random", "self"}, "Training mode must be random or self."
        # Allow for multiple modes. Random or self-trained available for now.
        """
        RL training loop. Plays num_games against random moves.
        """ 
        print(f"[TRAINING] Starting training for {num_games} games. | Mode = {mode}")

        wins = 0
        losses = 0
        draws = 0

        for game in range(num_games):
            board = np.zeros((6, 7), dtype=int)
            current = 1 # Player 1 starts
            opp_id = 3 - self.player_id
            done = False
            winner = None
            moves_played = 0

            while not done:
                valid_moves = self.get_valid_moves(board)

                # Agent moves
                if current == self.player_id:
                    # RL chooses epsilon greedy action
                    state = self.encode_state(board)

                    if random.random() < self.epsilon or state not in self.q_table:
                        action = random.choice(valid_moves)
                    else:
                        action = max(valid_moves, key=lambda a: self.q_table[state].get(a, 0.0))

                    # Simulate move
                    prev_board = board
                    next_board = self.simulate_move(board, action, self.player_id)
                    moves_played += 1 

                    # Check terminal condition after the move
                    if self.check_win(next_board, current):
                        winner = current
                        done = True
                    elif np.all(next_board[0] != 0): # Board is full, draw
                        winner = 0
                        done = True
                    
                    # Rewards
                    reward = self.local_reward(prev_board, next_board, action, current)

                    if done:
                        # Win / Loss / Draw rewards
                        if winner == self.player_id:
                            reward += 1.0
                            # Encourage faster wins (more remaining cells = more reward)
                            max_moves = self.rows * self.cols
                            remaining_cells = max_moves - moves_played
                            reward += 0.5 * (remaining_cells / max_moves)
                        elif winner == 0:
                            reward += 0.3 # Draw
                        else:
                            reward -= 1.0 # Loss
                    
                    next_state = self.encode_state(next_board)
                    self.update_q(state, action, reward, next_state, done)

                    # Update board
                    board = next_board
                
                # Opponent's move
                else:
                    # Selected mode = random
                    if mode == "random":
                        action = random.choice(valid_moves)
                    # Selected mode = self
                    elif mode == "self":
                        # Flip the board state so AI can play as both players
                        opp_state = self.encode_state(board, flip=True)
                        if random.random() < self.epsilon or opp_state not in self.q_table:
                            # Explore
                            action = random.choice(valid_moves)
                        else:
                            action = max(valid_moves, key=lambda a: self.q_table[opp_state].get(a, 0.0))
                    
                    # Simulate move and get next board
                    next_board = self.simulate_move(board, action, opp_id)
                    moves_played += 1
                
                    # Check whether move ended the game
                    if self.check_win(next_board, current):
                        winner = current
                        done = True
                    elif np.all(next_board[0] != 0): # Board is full, draw
                        winner = 0
                        done = True
                
                # Update board
                board = next_board
                # Switch players
                current = 2 if current == 1 else 1
            
            # Track stats to print and show progress
            if winner == self.player_id:
                wins += 1
            elif winner == 0:
                draws += 1
            else:
                losses += 1
            
            # Exploration decay
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # To print progress in 10 chunks
            if (game + 1) % max(1, num_games // 10) == 0:
                print(
                    f"[{game + 1}/{num_games}] "
                    f"ε={self.epsilon:.3f} | W/L/D = {wins}/{losses}/{draws}"
                )
            
        filename = f"qtables/qtable_{mode}_{num_games}.pkl"
        print(f"\n[TRAINING COMPLETE] Saving model → {filename}")
        self.save_q_table(filename)
    
    # Q-learning update rule
    def update_q(self, state, action, reward, next_state, done):

        # Initialize in q-table
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in range(self.cols)}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in range(self.cols)}

        # Initialize visitation counts
        if state not in self.visits:
            self.visits[state] = {a: 0 for a in range(self.cols)}
        
        # Increment visitation count for (state, action)
        self.visits[state][action] += 1
        n = self.visits[state][action]

        current_q = self.q_table[state][action]

        if done:
            max_next_state = 0.0
        else:
            max_next_state = max(self.q_table[next_state].values())

        target = reward + self.gamma * max_next_state

        # Visitation-based learning-rate decay
        effective_alpha = max(0.01, self.alpha / np.sqrt(n))
        new_q = current_q + effective_alpha * (target - current_q)
        self.q_table[state][action] = new_q


    # Flip flag allows us to "flip" the board state to see it from the other point of view,
    # helpful for self-training
    def encode_state(self, board, flip=False):
        # The "point of view" the AI will interpret the board in
        if flip:
            pov = 3 - self.player_id
        else:
            pov = self.player_id
        enc = []
        for value in board.flatten():
            if value == 0:
                enc.append(0)
            elif value == pov:
                enc.append(1)
            else:
                enc.append(-1)
        return tuple(enc)
    
    # For local reward shaping
    def local_reward(self, prev_board, next_board, col, player):
        """
        Compute a shaping reward based only on the consequences of this move.

        Includes:
        - Center column preference
        - Creating 2-in-a-row and 3-in-a-row
        - Blocking an opponent's win
        """
        # Our own moves only
        if player != self.player_id:
            return 0.0
        
        me = self.player_id
        opp = 3 - me
        reward = 0.0

        # Find the row the action was played ion
        row = self.find_drop_row(prev_board, col, player)
        if row is None:
            return 0.0
        
        # Center column preference normalized 0 to 0.1 reward
        center_col = self.cols // 2 # 3 for our implementation
        center_distance = abs(center_col - col)
        center_weight = (self.cols//2 - center_distance) / (self.cols//2)
        reward += 0.1 * center_weight

        # In-a-row checks
        lengths = []
        for dr, dc in [(0,1), (1,0), (1,1), (1,-1)]: # horizontal, vertical, diagonal
            length = self.line_length(next_board, row, col, me, dr, dc)
            lengths.append(length)
        max_len = max(lengths) if lengths else 1
        if max_len == 2:
            reward += 0.15
        elif max_len == 3:
            reward += 0.4
        
        # Blocking an immediate opponent win
        if self.blocked_win(prev_board, col, opp):
            reward += 0.6
        
        return reward

    # To find the row the action was played in
    def find_drop_row(self, prev_board, col, player):
        for r in range(self.rows - 1, -1, -1):
            if prev_board[r][col] == 0:
                return r
        return None
    
    # To count continous line length in a direction
    def line_length(self, board, row, col, player, dr, dc):
        length = 1

        # Check in the positive direction
        r, c = row + dr, col + dc
        while 0 <= r < self.rows and 0 <= c < self.cols and board[r][c] == player:
            length += 1
            r += dr
            c += dc

        # Check in the negative direction
        r, c = row - dr, col - dc
        while 0 <= r < self.rows and 0 <= c < self.cols and board[r][c] == player:
            length += 1
            r -= dr
            c -= dc

        return length
    
    # Simulate dropping a piece for 'player 'in 'col' on a copy of 'board', independent of simulate_move()
    def drop_piece(self, board, col, player):
        new_board = board.copy()
        for r in range(self.rows - 1, -1, -1):
            if new_board[r][col] == 0:
                new_board[r][col] = player
                break
        return new_board
    
    # To check if a move blocked an opponent's immediate win
    def blocked_win(self, prev_board, col, opp):
        # If opponent can't play here, no block
        if not self.is_valid_move(prev_board, col):
            return False
        
        # Before move: would opponent win?
        before = self.drop_piece(prev_board, col, opp)
        if not self.check_win(before, opp):
            return False
        
        # After move: can they still win?
        if not self.is_valid_move(before, col):
            return True # Column is full, so we blocked it
        
        return True

    # To save a q table to a given path
    def save_q_table(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

    # To load a q table from a given path
    def load_q_table(self, path: str):
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)
