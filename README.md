# Q-Learning Pathfinding and Analysis Project

## Overview

This project implements a **Q-Learning** algorithm for solving pathfinding tasks in a custom grid environment but use a advance data structure like a  **Graph** in order to explore a save the experience of agents. The agent learns how to navigate the environment to complete tasks like picking up and dropping off items while avoiding obstacles. Additionally, this project offers tools for **deep analysis** of the Q-Learning process, allowing users to analyze, visualize, and evaluate the agent’s decision-making and reward patterns.

The project includes utilities for:
- **Training** the Q-Learning agent with various hyperparameters.
- **Saving and loading** Q-tables for future predictions.
- **Visualizing** training rewards and performance metrics.
- **Analyzing** the learned Q-table for decision-making insights using 2D and 3D plots.
- **Evaluating** prediction accuracy and the paths taken by the agent.

## Features

- **Custom Grid Environment**: The agent navigates a grid, performing tasks like item pickup and drop-off while avoiding blocked zones.
- **Q-Learning Algorithm**: Implements Q-Learning for training the agent over a customizable number of iterations.
- **Reward Visualization**: Displays rewards over time to monitor the agent's learning progress.
- **Q-Table Management**: Save and load Q-tables for training continuation or prediction. This Qtable is a Graph.
- **Advanced Path Analysis**: Analyze multiple path outcomes, including probability distributions and performance metrics.
- **Graphical Outputs**: 2D and 3D scatter plots of reward and path analysis for comprehensive visualization of results.

## Files

- **`QLearning.py`**: Core implementation of the Q-Learning algorithm. This file manages the agent’s training, Q-table updates, saving/loading models, and hyperparameter tuning.
- **`Games.py`**: Defines the grid environment, including the setup for zones, obstacles, item locations, and the initial position of the agent.
- **`DeepAnalysis.py`**: Contains functions for analyzing the Q-Learning agent’s behavior, path predictions, and reward distribution across multiple iterations.
- **`Q_Graphs.py`**: Handles the Data Structure Graph representation of the agent’s performance.

- **`main.py`**: The main entry point for running the training, predictions, and analysis of the Q-Learning agent. This script allows for customization of grid size, hyperparameters, and visualization settings.

## How to Run

1. **Install Requirements**:

   Ensure the required dependencies are installed by using:
   ```bash
   pip install -r requirements.txt

2.	**Train the Q-Learning Agent**:
   You can initiate the training process using the main.py script. By default, the agent trains on a 10x10 grid with predefined item locations and blocked zones. You can adjust the hyperparameters such as the number of iterations, epsilon, and gamma within the script.

3.	**Visualize Rewards**:
   After training, the project outputs visualizations of the rewards, allowing you to see how the agent’s performance improves over time.
4.	**Q-Table Analysis**:
   Run the Q-table analysis through main.py to analyze the agent’s decision-making process:
5.	**Load and Predict**:
   You can use a saved Q-table to predict outcomes in the environment without retraining:

## Hyperparameters

The following hyperparameters can be customized in main.py or in QLearning.py:

	•	gamma: Discount factor for future rewards.
	•	epsilon: Exploration-exploitation tradeoff parameter.
	•	alpha: Learning rate for updating Q-values.

## Output

	•	Reward Plot: Displays the rewards the agent accumulates over the course of training.
	•	Q-Table File: The Q-table is saved as a .joblib file for later use.
	•	Scatter Plots: 2D and 3D scatter plots visualizing the agent’s performance, including steps, rewards, and path analysis.

## Contributing

Contributions to improve the Q-Learning agent, add features, or extend the environment are welcome. Feel free to submit pull requests or raise issues.

License

This project is licensed under the MIT License - see the LICENSE file for details.
