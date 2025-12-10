import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from typing import List, Tuple, Optional
from collections import deque
from .base_ai import Connect4AI
from models import HeuristicAI
from models.minimax_ab_ai import MinimaxABAI

class Connect4CNN(nn.Module):
    """
    Small convolutional network that maps a 2x6x7 board to 7 Q-values (one per column)
    """
    def __init__(self):
        super().__init__()

        # Input channels = 2 (our pieces and opponent pieces)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=4, padding=1) # (64, 5, 6)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # (128, 5, 6)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # (128, 5, 6)

        # Flatten to a fully connected layer: 128 * 5 * 6 = 3840
        self.fc1 = nn.Linear(128 * 5 * 6, 256)
        self.fc2 = nn.Linear(256, 7) # Output Q-values for each of the 7 columns

    def forward(self, x):
        """
        Forward pass through the network.
        """
        # x shape: (batch_size, 2, 6, 7)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        x = torch.flatten(x, start_dim=1)  # Flatten all but the batch
        x = torch.relu(self.fc1(x))
        q_values = self.fc2(x)
        return q_values

# CNN based Reinforcement Learning AI Agent
class CNNRLAI(Connect4AI):
    """
    CNN based Deep Q-Learning AI for Connect 4
    """
    def __init__(self,
                 player_id: int,
                 gamma: float = 0.99,
                 lr: float = 0.0005,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.999995,
                 epsilon_min: float = 0.05,
                 replay_size: int = 20000,
                 batch_size: int = 32,
                 target_update_freq: int = 500,
                 min_replay_size: int = 1000,
                 device: Optional[str] = None
                 ):
        super().__init__(player_id)

        # Board dimensions
        self.rows = 6
        self.cols = 7

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.min_replay_size = min_replay_size

        # Other models
        self.heuristic_opponent = HeuristicAI(player_id=3 - self.player_id)
        self.minimax_opponent = MinimaxABAI(player_id=3 - self.player_id, depth=5)

        # Device configuration
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Policy and target networks
        self.policy_network = Connect4CNN().to(self.device)
        self.target_network = Connect4CNN().to(self.device)

        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval() # Target network in eval mode

        # Optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

        # Experience replay buffer
        self.memory = deque(maxlen=replay_size)

        # To track training steps to know when to update target network
        self.training_steps = 0

        # For logging / tracking
        self.episode_rewards = []
        self.moving_avg_rewards = []

    def encode_board_np(self, board: np.ndarray, pov_player: Optional[int] = None) -> torch.Tensor:
        """
        Encode board as a (2, 6, 7) numpy array of floats

        Channel 0: 1 where pov_player has pieces, 0 elsewhere
        Channel 1: 1 where opponent has pieces, 0 elsewhere
        """
        # Determine point of view player
        if pov_player is None:
            pov_player = self.player_id
        opp_id = 3 - pov_player

        player_plane = (board == pov_player).astype(np.float32)
        opp_plane = (board == opp_id).astype(np.float32)
        stacked_state = np.stack([player_plane, opp_plane], axis=0) # (2, 6, 7)
        return stacked_state
    
    def encode_board(self, board: np.ndarray, pov_player: Optional[int] = None) -> torch.Tensor:
        """
        Torch version of encode_board_np
        """
        arr = self.encode_board_np(board, pov_player=pov_player)
        return torch.tensor(arr, dtype=torch.float32)
    
    def simulate_move(self, board: np.ndarray, col: int, player: int) -> np.ndarray:
        """
        Simulate dropping a piece in the specified column for the given player.
        Returns the new board state.
        """
        new_board = board.copy()
        for row in range(self.rows - 1, -1, -1):
            if new_board[row, col] == 0:
                new_board[row, col] = player
                break
        return new_board
    
    def find_drop_row(self, board: np.ndarray, col:int) -> Optional[int]:
        """
        Find the row index where a piece would land if dropped in the specified column.
        Returns None if the column is full.
        """
        for row in range(self.rows - 1, -1, -1):
            if board[row, col] == 0:
                return row
        return None
    
    def line_length(self, board: np.ndarray, row: int, col: int, player: int, dr: int, dc: int) -> int:
        """
        Calculate the length of a continuous line of pieces for the given player
        """
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
    
    def drop_piece(self, board: np.ndarray, col: int, player: int) -> np.ndarray:
        """
        Return a new board state after the player drops a piece into col.
        Used to check for hypothetical blocks or wins.
        """
        new_board = board.copy()
        for r in range(self.rows - 1, -1, -1):
            if new_board[r][col] == 0:
                new_board[r][col] = player
                break
        return new_board
    
    
    def blocked_win(self, prev_board: np.ndarray, col: int, opp: int) -> bool:
        """
        Check if dropping a piece in col blocks an opponent's immediate win.
        """
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
    
    def local_reward(self, prev_board: np.ndarray, next_board: np.ndarray, col: int, row: int) -> float:
        """
        Reward shaping for a single move:
        - Center column preference
        - Creating of lines/pieces in a row
        - Blocking opponent's immediate win
        """
        me = self.player_id
        opp = 3 - me
        reward = 0.0
        if row is None:
            return 0.0
        
        # Center column preference
        center_col = self.cols // 2
        center_distance = abs(center_col - col)
        center_weight = max(0.0, (center_col - center_distance) / center_col)
        reward += 0.1 * center_weight

        # Check for pieces in a row (horizontal, vertical, and diagonal)
        lengths = []
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            lengths.append(self.line_length(next_board, row, col, me, dr, dc))
        max_length = max(lengths) if lengths else 1

        if max_length == 2: # 2 in a row
            reward += 0.15
        elif max_length == 3: # 3 in a row
            reward += 0.4
        
        # Blocking opponent's immediate win
        if self.blocked_win(prev_board, col, opp):
            reward += 0.6
        
        return reward
    
    def select_action_training(self, board: np.ndarray) -> int:
        """
        Select action using epsilon-greedy strategy during training.
        """
        valid_moves = self.get_valid_moves(board)

        # Explore
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        # Exploit
        state_tensor = self.encode_board(board).unsqueeze(0).to(self.device)  # (1, 2, 6, 7)
        with torch.inference_mode():
            q_values = self.policy_network(state_tensor)[0].cpu().numpy()
        
        # Restrict to valid moves
        best_move = max(valid_moves, key=lambda col: q_values[col])
        return best_move
    
    def select_action_selfplay(self, board: np.ndarray) -> int:
        """
        Select an action for the opponent using the same Q-network,
        but from the opponent's perspective.
        """
        opp_id = 3 - self.player_id
        valid_moves = self.get_valid_moves(board)

        # Exploration for opponent too
        if random.random() < self.epsilon:
            return random.choice(valid_moves)

        # Encode from opponent POV
        state_np = self.encode_board_np(board)
        opp_state_np = self.invert_pov(state_np)
        state_tensor = torch.tensor(opp_state_np, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            q_values = self.policy_network(state_tensor)[0].cpu().numpy()

        best_move = max(valid_moves, key=lambda col: q_values[col])
        return best_move

    
    def store_transition(self,
                         state_np: np.ndarray,
                         action: int,
                         reward: float,
                         next_state_np: np.ndarray,
                         done: bool):
        """
        Storte a transition in the replay buffer.
        States are stored as numpy arrays, not tensors, for effifciency.
        """
        self.memory.append((state_np, action, reward, next_state_np, done))

    def valid_mask_from_states(self, states_np: np.ndarray) -> np.ndarray:
        """
        Compute a mask of valid actions for each state in the batch.
        states_np: (batch_size, 2, 6, 7)
        A column is invalid if the top row is occupied in either channel.
        """
        # Channel 0 + Channel 1 > 0 = there is a piece
        # Shape: (barch_size, 6, 7)
        occupancy = states_np[:, 0, :, :] + states_np[:, 1, :, :]
        top_row = occupancy[:, 0, :] # (batch_size, 7)
        valid_mask = (top_row == 0).astype(np.float32) # 1 where valid, 0 where full
        return valid_mask

    def train_step(self):
        """
        Sample a batch from the replay buffer and perform one gradient step.
        """
        if len(self.memory) < max(self.batch_size, self.min_replay_size):
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states_np, actions, rewards, next_states_np, dones = zip(*batch)

        states_np = np.stack(states_np, axis=0) # (batch_size, 2, 6, 7)
        next_states_np = np.stack(next_states_np, 0) # (batch_size, 2, 6, 7)

        # Build valid action masks for next states
        next_valid_mask_np = self.valid_mask_from_states(next_states_np) # (batch_size, 7)

        states = torch.from_numpy(states_np).float().to(self.device) # (batch_size, 2, 6, 7)
        next_states = torch.from_numpy(next_states_np).float().to(self.device) # (batch_size, 2, 6, 7)
        next_valid_mask = torch.from_numpy(next_valid_mask_np).float().to(self.device) # (batch_size, 7)

        actions = torch.tensor(actions, dtype=torch.long, device=self.device) # (batch_size, )
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device) # (batch_size, )
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device) # (batch_size, )

        # Current Q-values
        q_values = self.policy_network(states) # (batch_size, 7)
        q_actions = q_values.gather(1, actions.unsqueeze(1)).squeeze(1) # (batch_size, )

        # Target Q-values
        # r + gamma * max_a' Q_target(s', a') if not done
        with torch.no_grad():
            next_q_values = self.target_network(next_states) # (batch_size, 7)

            # Mask invalid actions by setting their Q-values to a very low number so max() ignores them
            invalid_mask = (next_valid_mask == 0.0)
            next_q_values = next_q_values.masked_fill(invalid_mask, -1e4)

            max_next_q, _ = torch.max(next_q_values, dim=1) # (batch_size, )
        
            target = rewards + (1.0 - dones) * self.gamma * max_next_q
            target = target.to(q_actions.dtype)

        # Loss : Huber Loss
        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(q_actions, target)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_steps += 1

        # Update target network periodically
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())
        
    def get_move(self, board: np.ndarray) -> int:
        """
        Used by the game (demo.py): choose the best move (no exploration)
        """
        valid_moves = self.get_valid_moves(board)

        state_tensor = self.encode_board(board).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            q_values = self.policy_network(state_tensor)[0].cpu().numpy()
        
        best_move = max(valid_moves, key=lambda col: q_values[col])
        return best_move
    
    def invert_pov(self, state_np: np.ndarray) -> np.ndarray:
        """
        Invert the point of view of the state numpy array.
        """
        return np.stack([state_np[1], state_np[0]], axis=0)
    
    def train(self,
              num_games: int = 10000,
              mode: str = "random",
              print_chunks: int = 100):
        """
        Train the CNN RL agent with a self-contained simulation.
        
        Mode:
        - "random": opponent plays uniformly random moves
        - "self": opponent plays using the same Q-network (self-play)
        - "heuristic": opponent plays using HeuristicAI
        
        Each episode:
        - Our agent moves (epsilon-greedy)
        - Opponent moves
        - Reward is shaping + terminal
        - Each step yields a DQN transition (s, a, r, s', done)
        """

        print(f"Training CNN RL for {num_games} games | Mode = {mode} | Epsilon Decay = {self.epsilon_decay} | Device = {self.device}")

        wins = 0
        losses = 0
        draws = 0
        opp_id = 3 - self.player_id
        # Chunk stats for progress printing
        chunk_wins = 0
        chunk_losses = 0
        chunk_draws = 0

        for game in range(num_games):
            board = np.zeros((self.rows, self.cols), dtype=int)
            done = False
            winner = None
            moves_played = 0
            episode_reward = 0.0

            while not done:
                # RL agent's turn
                state_np = self.encode_board_np(board)  # For replay
                state_tensor = torch.from_numpy(state_np).float().to(self.device)

                # Choose action for training
                action = self.select_action_training(board)
                prev_board = board.copy()
                row = self.find_drop_row(prev_board, action)
                board_after_our_move = self.simulate_move(board, action, self.player_id)
                moves_played += 1

                # Shaping reward from just our move
                
                reward = self.local_reward(prev_board, board_after_our_move, action, row)

                # Check for win
                if self.check_win(board_after_our_move, self.player_id):
                    winner = self.player_id
                    done = True
                    reward += 2.0 # Win reward
                    max_moves = self.rows * self.cols # 42
                    remaining_cells = max(0, max_moves - moves_played)
                    reward += 0.5 * (remaining_cells / max_moves) # Reward winning with less moves

                    next_state_np = self.encode_board_np(board_after_our_move)
                    self.store_transition(state_np, action, reward, next_state_np, True)
                    self.train_step()
                    episode_reward += reward
                    board = board_after_our_move
                    break

                # Board full after our move = draw
                if np.all(board_after_our_move[0] != 0):
                    winner = 0
                    done = True
                    reward += 0.5 # Draw reward
                    next_state_np = self.encode_board_np(board_after_our_move)
                    self.store_transition(state_np, action, reward, next_state_np, True)
                    self.train_step()
                    episode_reward += reward
                    board = board_after_our_move
                    break
                
                # Opponent's turn
                if mode == "random":
                    opp_valid_moves = self.get_valid_moves(board_after_our_move)
                    opp_action = random.choice(opp_valid_moves)
                elif mode == "self":
                    opp_action = self.select_action_selfplay(board_after_our_move)
                elif mode == "heuristic":
                    opp_action = self.heuristic_opponent.get_move(board_after_our_move)
                else:
                    raise NotImplementedError("Selected mode not implemented for CNN-RL")
                board_after_opp_move = self.simulate_move(board_after_our_move, opp_action, opp_id)
                moves_played += 1

                # Check if opponent won
                if self.check_win(board_after_opp_move, opp_id):
                    winner = opp_id
                    done = True
                    reward -= 2.0  # Loss penalty
                    next_state_np = self.encode_board_np(board_after_opp_move)
                    self.store_transition(state_np, action, reward, next_state_np, True)
                    self.train_step()
                    episode_reward += reward
                    board = board_after_opp_move
                    break

                # Board full after opponent's move = draw
                if np.all(board_after_opp_move[0] != 0):
                    winner = 0
                    done = True
                    reward += 0.3
                    next_state_np = self.encode_board_np(board_after_opp_move)
                    self.store_transition(state_np, action, reward, next_state_np, True)
                    self.train_step()
                    episode_reward += reward
                    board = board_after_opp_move
                    break

                # Otherwise, continue game
                next_state_np = self.encode_board_np(board_after_opp_move)
                self.store_transition(state_np, action, reward, next_state_np, False)
                self.train_step()
                episode_reward += reward
                board = board_after_opp_move
            
            # Game over, update stats and decay epsilon
            if winner == self.player_id:
                wins += 1
                chunk_wins += 1
            elif winner == 0:
                draws += 1
                chunk_draws += 1
            else:
                losses += 1
                chunk_losses += 1
            
            # Update reward tracking
            self.update_reward_tracking(episode_reward)

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Progress print
            chunk = max(1, num_games // print_chunks)
            chunk_total = chunk_wins + chunk_losses + chunk_draws
            chunk_win_rate = chunk_wins / chunk_total if chunk_total > 0 else 0.0
            if (game + 1) % chunk == 0:
                print(
                    f"[{game + 1}/{num_games}] "
                    f"Epsilon = {self.epsilon:.3f} | "
                    f"W/L/D = {wins}/{losses}/{draws} | "
                    f"Avg Reward (last 100) = {self.moving_avg_rewards[-1]:.3f} | "
                    f"Chunk Win Rate = {chunk_win_rate:.3f}"
                )
                chunk_wins = 0
                chunk_losses = 0
                chunk_draws = 0
            
        print(f"\n[TRAINING COMPLETE] CNN RL finished {num_games} games | Mode: {mode} | Epsilon decay: {self.epsilon_decay}")
        print(f"Final stats: W={wins}, L={losses}, D={draws}")
        self.save_training_history(f"training_history/training_{num_games}_{mode}_{self.epsilon_decay}.npz")

    def update_reward_tracking(self, episode_reward: float):
        """
        Track raw reward, moving avg.
        """
        # Store raw episode reward
        self.episode_rewards.append(episode_reward)

        # Compute 100 episode moving average
        window = 100
        if len(self.episode_rewards) >= window:
            avg = sum(self.episode_rewards[-window:]) / window
        else:
            avg = sum(self.episode_rewards) / len(self.episode_rewards)
        
        self.moving_avg_rewards.append(avg)

    def has_nan_weights(self):
        """
        Check if any weights in the policy network are NaN.
        """
        return any(torch.isnan(p).any() for p in self.policy_network.parameters())

    def save_training_history(self, path: str):
        """
        Save training reward history to a .npz file for plotting later
        """
        np.savez(path,
                episode_rewards=np.array(self.episode_rewards),
                moving_avg_rewards=np.array(self.moving_avg_rewards))
        print(f"Saved training history to {path}")

    
    # Helpers for saving and loading policy networks
    def save_model(self, path: str):
        """
        Save the CNN policy network to specified path
        """
        if self.has_nan_weights():
            print("WARNING: Model contains NaN weights — NOT SAVING")
            return
        torch.save(self.policy_network.state_dict(), path)
        print(f"Saved model to {path}")
    
    def load_model(self, path: str):
        """
        Load the CNN policy network from specified path
        """
        state_dict = torch.load(path, map_location=self.device)
        self.policy_network.load_state_dict(state_dict)
        self.target_network.load_state_dict(state_dict)