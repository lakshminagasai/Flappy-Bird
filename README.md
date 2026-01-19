# Using Deep Q-Network to Learn How to Play Flappy Bird 🎮
<img src="DeepLearningFlappyBird/images/flappy_bird_demp.gif" width="250">

## Overview

This project implements a **Deep Q-Network (DQN)** to train an agent to play the Flappy Bird game using **raw pixel inputs**.
The implementation is inspired by the seminal papers *Playing Atari with Deep Reinforcement Learning* and *Human-level Control through Deep Reinforcement Learning*, demonstrating that DQN can be generalized beyond Atari environments to Flappy Bird.


## Installation Dependencies

*  Python 3.10
* TensorFlow 2.x
* pygame
* OpenCV (opencv-python)-

## How to Run

```bash
git clone https://github.com/lakshminagasai/Flappy-Bird.git
cd DeepLearningFlappyBird
python deep_q_network.py
```
---

## What is a Deep Q-Network?

A **Deep Q-Network (DQN)** is a convolutional neural network trained using a variant of **Q-learning**.

* **Input:** Raw pixel frames from the game screen
* **Output:** Q-values representing the expected future reward for each possible action

The agent selects actions using an **ε-greedy policy**, balancing exploration and exploitation.

---

## Deep Q-Network Algorithm

The pseudo-code for the Deep Q-Learning algorithm (from [1]) is shown below:

```text
Initialize replay memory D to capacity N
Initialize action-value function Q with random weights θ

for episode = 1 to M do
    Initialize state s₁
    for t = 1 to T do
        With probability ε select a random action aₜ
        otherwise select aₜ = argmaxₐ Q(sₜ, a; θ)

        Execute action aₜ and observe reward rₜ and next state sₜ₊₁
        Store transition (sₜ, aₜ, rₜ, sₜ₊₁) in D

        Sample a minibatch from replay memory D
        Compute target:
            yⱼ = rⱼ (if terminal)
            yⱼ = rⱼ + γ maxₐ′ Q(sⱼ₊₁, a′; θ) (if non-terminal)

        Perform gradient descent on (yⱼ − Q(sⱼ, aⱼ; θ))²
    end for
end for
```

---

## Experiments

### Environment

The DQN is trained directly on **pixel-level observations**.
To speed up convergence, the game background is removed, following the approach in [3].

### Network Architecture

The preprocessing pipeline:

1. Convert frames to grayscale
2. Resize frames to **80 × 80**
3. Stack the last **4 frames** to form an **80 × 80 × 4** input

**Network Structure:**

* Conv Layer 1: 8×8 kernel, 32 filters, stride 4 → Max Pool
* Conv Layer 2: 4×4 kernel, 64 filters, stride 2 → Max Pool
* Conv Layer 3: 3×3 kernel, 64 filters, stride 1 → Max Pool
* Fully Connected Layer: 256 ReLU units
* Output Layer: Q-values for each valid action

At each time step, the agent selects the action with the highest Q-value using an ε-greedy strategy.

---

## References

[1] Mnih et al., **Human-level Control through Deep Reinforcement Learning**, *Nature*, 2015.

[2] Mnih et al., **Playing Atari with Deep Reinforcement Learning**, *NIPS Deep Learning Workshop*.

[3] Kevin Chen, **Deep Reinforcement Learning for Flappy Bird**

* [Project Report](http://cs229.stanford.edu/proj2015/362_report.pdf)
* [YouTube Result](https://youtu.be/9WKBzTUsPKc)
