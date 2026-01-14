# Deep Q-Network AI for Flappy Bird

<img src="./images/flappy_bird_demp.gif" width="250">

Train an AI agent to play **Flappy Bird** using **Deep Q-Learning**. This project demonstrates how a convolutional neural network can learn to play a game directly from pixel input using reinforcement learning.

---

## Overview

This project implements a **Deep Q-Network (DQN)** to train an AI agent to play Flappy Bird. The agent learns by:

1. Observing the game screen (state)  
2. Making decisions (flap or do nothing)  
3. Receiving rewards based on survival and passing pipes  
4. Updating its action-value estimates using a neural network  

The model learns to maximize its total game score over time without any hard-coded rules.

---

## Dependencies

- Python 3.x (recommended)  
- TensorFlow 0.7 (or compatible version)  
- pygame  
- OpenCV-Python  

Install dependencies using pip:

```bash
pip install tensorflow pygame opencv-python
How to Run
bash
Copy code
# Clone the repository
git clone https://github.com/lakshminagasai/Flappy-Bird.git
cd Flappy-Bird/DeepLearningFlappyBird

# Run the AI agent
python deep_q_network.py
Note: Do not include the virtual environment (flappy310/) in the repository. It is excluded in .gitignore.

Deep Q-Network Overview
A Deep Q-Network (DQN) approximates the Q-function:

css
Copy code
Q(s, a) ≈ expected future rewards given state s and action a
Input: stacked grayscale frames of the game (80x80x4)

Architecture: convolutional layers followed by fully connected layers

Output: Q-values for each possible action (flap / do nothing)

DQN Training Algorithm
Initialize replay memory and Q-network with random weights.

For each episode:

Choose actions using an ϵ-greedy policy

Store transitions (state, action, reward, next_state) in replay memory

Sample minibatches of experiences to update the network using:

ini
Copy code
Loss = (r + γ * max_a' Q(next_state, a') - Q(state, action))^2
Linearly anneal ϵ from 0.1 → 0.0001 to reduce random exploration over time.

Continue training to improve the agent’s performance.

Preprocessing
Convert frames to grayscale

Resize to 80x80 pixels

Stack the last 4 frames to provide temporal information

Remove the background to speed up convergence

Network Architecture
Conv Layer 1: 8x8 kernel, 32 filters, stride 4 + 2x2 max pooling

Conv Layer 2: 4x4 kernel, 64 filters, stride 2 + 2x2 max pooling

Conv Layer 3: 3x3 kernel, 64 filters, stride 1 + 2x2 max pooling

Fully Connected Layer: 256 ReLU nodes

Output Layer: Q-values for each valid action

<img src="./images/network.png" width="450">
At each time step, the network selects the action with the highest Q-value, following an ϵ-greedy policy.

Folder Structure
bash
Copy code
Flappy-Bird/
├─ DeepLearningFlappyBird/   # Project code
├─ images/                    # Demo images/GIFs
├─ README.md                  # Project documentation
├─ .gitignore                 # Ignored files/folders
Virtual environment (flappy310/) and saved networks (saved_networks/) are ignored.

Notes
Ensure dependencies are installed in a separate virtual environment.

You can export all dependencies for reproducibility:

bash
Copy code
pip freeze > requirements.txt
To recreate the environment:

bash
Copy code
python -m venv env
source env/bin/activate   # Linux/macOS
env\Scripts\activate      # Windows
pip install -r requirements.txt
Author
Lakshmi Naga Sai – Designed and implemented this Flappy Bird DQN AI project.
