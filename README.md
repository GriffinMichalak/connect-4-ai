# Connect 4 AI Agents

Connect-4 is a two-player board game in which players take turns placing chips into a 7 row × 6 column board until a player wins. Wins occur when a player has aligned four consecutive pieces vertically, horizontally, or diagonally. By utilizing Heuristics, Minimax, Q-learning, and Monte Carlo Tree Search, we aim to build and evaluate four models that consistently outperform human players

## 🎯 Project Overview
This project implements and compares various AI algorithms to play the populate board game Connect 4. Here, we provide a platform to understand the pros adn cons of different AI models to solve the game. To demonstrate our models, we built a Graphical UI with Python's Pygame. We support the following three modes: human vs human, human vs AI, and AI vs AI

## ✨ Features

- **Playable**: UI developed in Pygame for playing Connect 4
- **Multiple AI Algorithms**: Compare different AI strategies side-by-side
- **Game Modes**: Play against AI, watch AI vs AI, Human vs Human
- **Pre-trained Models**: Includes trained reinforcement learning models OOTB
- **Evaluation Tools**: Python scripts used to analyze and comapre our models

## 🤖 Models

### 1. **Minimax Algorithm**
A classical game-theoretic approach that examines the game tree to identify optimal moves. Uses depth-limited search combined with evaluation functions

### 2. **Minimax with Alpha-Beta Pruning**
Optimized version of Minimax that reduces the number of nodes evaluated by pruning branches that cannot affect the final decision

### 3. **Monte Carlo Tree Search (MCTS)**
Probabilistic algorithm that uses random simulations to evaluate moves, balancing exploration and exploitation

### 4. **Heuristic Model**

Rule-based approach using domain-specific heuristics such as:
- Center column preference
- Threat detection (blocking opponent wins)
- Sequence counting (2-in-a-row, 3-in-a-row)
- Potential win opportunities

### 5. **Reinforcement Learning (CNN-based)**

Deep learning approach using Convolutional Neural Networks trained through reinforcement learning. Multiple pre-trained models are available:
- Models trained against random opponents
- Models trained against heuristic opponents
- Models trained through self-play

### 6. **Random AI**

Baseline opponent that makes random valid moves for comparison purposes

## 📋 Requirements (Specified in `requirements.txt`)
- Python 3.7+
- pygame >= 2.0.0
- numpy >= 1.20.0
- matplotlib
- torch (PyTorch)

## 🚀 Installation
### 1. Clone the Repository
```sh
git clone https://github.com/GriffinMichalak/connect-4-ai
cd connect-4-ai
```

### 2. Create Virtual Environment
```sh
python3 -m venv .venv
# or, depending on python version
python -m venv .venv
```

### 3. Activate Virtual Environment

**On macOS/Linux:**
```sh
source .venv/bin/activate
```
**On Windows:**
```sh
.venv\Scripts\activate
```
### 4. Install Dependencies
```sh
pip install -r requirements.txt
```
## 🎮 Usage

### Running the Demo

Start the interactive demo:
```sh
python demo.py
```
The demo provides three main game modes:

1. **Human vs Human**: Two players take turns on the same computer
2. **Human vs AI**: Play against one of the available AI algorithms
3. **AI vs AI**: Watch two AI algorithms compete against each other

### Selecting an AI Opponent
When choosing "Human vs AI", you'll be presented with the following options:
- **0. Random Selection**: Baseline random move AI
- **1. Minimax (Basic)**: Standard minimax algorithm
- **2. Minimax w/ Alpha-Beta Pruning**: Optimized minimax
- **3. Reinforcement Learning**: CNN-based RL model (requires selecting a pre-trained model)
- **4. Simple Heuristic Search**: Rule-based heuristic AI
- **5. Monte Carlo Tree Search**: MCTS algorithm

### Training Reinforcement Learning Models

To train your own RL models:
```sh
python train_rl.py
```

Trained models will be saved in the `CNN_models/` directory

### Evaluating AI Performance
Use the evaluation scripts to analyze AI performance:
```sh
python evaluate_rl.py
python evaluate_heatmap.py
```
## 🔬 Algorithm Comparison
Each AI algorithm has different characteristics:
- **Minimax/Alpha-Beta**: Optimal play but computationally expensive
- **MCTS**: Good balance between performance and computation
- **Heuristic**: Fast and interpretable, but limited by hand-crafted rules
- **RL (CNN)**: Learns from experience, can discover novel strategies

## 📊 Pre-trained Models


The project includes several pre-trained CNN models:

- **Random-trained models**: Trained against random opponents (100k and 500k iterations)
- **Heuristic-trained models**: Trained against heuristic AI (100k and 500k iterations)
- **Self-play models**: Trained through self-play (100k and 500k iterations)
Model filenames indicate training method and iteration count
