"""
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
            epsilon_decay: float = 0.999997,
            epsilon_min: float = 0.05,
            q_table_path: Optional[str] = None
        ):
        super().__init__(player_id)

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
        """
        RL training loop. Plays num_games against random moves.
        """ 
        print(f"[TRAINING] Starting training for {num_games} games vs. random AI")

        wins = 0
        losses = 0
        draws = 0

        for game in range(num_games):
            board = np.zeros((6, 7), dtype=int)
            current = 1 # Player 1 starts
            last_state = None
            last_action = None
            done = False

            while not done:
                valid_moves = self.get_valid_moves(board)

                if current == self.player_id:
                    # RL chooses epsilon greedy action
                    state = self.encode_state(board)

                    if random.random() < self.epsilon or state not in self.q_table:
                        action = random.choice(valid_moves)
                    else:
                        action = max(valid_moves, key=lambda a: self.q_table[state].get(a, 0.0))

                    # Simulate move
                    next_board = self.simulate_move(board, action)
                    last_state = state
                    last_action = action
                
                else:
                    # Let opponent move (in this case randomly)
                    opp_move = random.choice(valid_moves)
                    next_board = self.simulate_move(board, opp_move)
                
                # Check whether move ended the game
                if self.check_win(next_board, current):
                    winner = current
                    done = True
                elif np.all(next_board[0] != 0): # Board is full, draw
                    winner = 0
                    done = True
                else:
                    winner = None # No winner yet, continue
                
                board = next_board
                current = 2 if current == 1 else 1 # Next player's turn
            
            # Rewards
            if last_state is not None:
                reward = (
                    +1 if winner == self.player_id else                 # +1 if win
                    -1 if winner not in (self.player_id, None, 0) else  # -1 if lose
                    0.3                                                 # 0.3 if draw
                )
                next_state = self.encode_state(board)
                self.update_q(last_state, last_action, reward, next_state)
            
            # Exploration decay
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # To print progress in 10 chunks
            if (game + 1) % max(1, num_games // 10) == 0:
                print(f"[{game + 1}/{num_games}] ε={self.epsilon:.3f}")
            
        filename = f"qtables/qtable_random_{num_games}.pkl"
        print(f"\n[TRAINING COMPLETE] Saving model → {filename}")
        self.save_q_table(filename)
    
    # Q-learning update rule
    def update_q(self, state, action, reward, next_state):

        # Initialize in q-table
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in range(7)}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in range(7)}

        current_q = self.q_table[state][action]
        max_next_state = max(self.q_table[next_state].values())

        new_q = current_q + self.alpha * (reward + self.gamma * max_next_state - current_q)
        self.q_table[state][action] = new_q


    def encode_state(self, board):
        enc = []
        for value in board.flatten():
            if value == 0:
                enc.append(0)
            elif value == self.player_id:
                enc.append(1)
            else:
                enc.append(-1)
        return tuple(enc)

    def save_q_table(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, path: str):
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)