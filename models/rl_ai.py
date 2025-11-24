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
            alpha: float = 0.5, 
            gamma: float = 0.95,
            epsilon: float = 1.0,
            epsilon_decay: float = 0.9995,
            epsilon_min: float = 0.05,
            q_table_path: Optional[str] = None
        ):
        super().__init__(player_id)

        # Q-table: state (flattened tuple) -> {actiin: Q-value}
        self.q_table: Dict[Tuple[int, ...], Dict[int, float]] = {}

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