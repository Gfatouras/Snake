import random
import pickle  # To save the model
import matplotlib.pyplot as plt
import sys
import numpy as np  # For calculating moving average
from collections import deque  # For efficient memory buffer
from utils import (
    GRID_SIZE, ACTIONS, directions, get_state,
    spawn_food, move_snake, manhattan_distance
)

# Hyperparameters - OPTIMIZED
ALPHA = 0.025     # Learning rate (reduced for more stable convergence)
GAMMA = 0.97     # Discount factor (increased to value long-term rewards)
EPSILON = 1.0    # Initial exploration rate
EPSILON_DECAY = 0.999   # Decay rate (faster decay - reaches min around game 1000)
EPSILON_MIN = 0.001      # Minimum exploration rate (keep small exploration)

MEMORY_SIZE = 2048      # Memory buffer size (increased for better sampling)
BATCH_SIZE = 64         # Batch size for training (reduced for faster early training)
REPLAY_FREQUENCY = 4    # Only replay every N steps (optimization)
NUM_GAMES = 8_000       # Number of iterations

MOVING_AVERAGE_CHART_VAL = 100  # Last 100 games average

# Rewards & Penalties - OPTIMIZED
REWARD_FOOD = 10.0          # Reward for eating food (increased)
PENALTY_DEATH = -10.0       # Penalty for dying (wall or tail)
PENALTY_MOVE = -0.01        # Small penalty for each movement (reduced)
REWARD_CLOSER = 0.1         # Small reward for getting closer to food
PENALTY_FARTHER = -0.1      # Small penalty for getting farther from food

# Game parameters
MAX_STEPS_WITHOUT_FOOD = 100  # Prevent infinite loops
initial_snake = [(5, 5)]  # Snake starts at the center of the grid

# Initialize Q-table and Memory buffer - OPTIMIZED
Q_table = {}
memory = deque(maxlen=MEMORY_SIZE)  # Use deque for O(1) operations
step_counter = 0  # Global step counter for replay timing

# Lists for plotting
epsilons = []
scores = []            # Individual game scores
cumulative_scores = [] # Cumulative sum of scores over time
iterations = []
steps_per_game = []    # Track steps taken per game

# Epsilon-greedy action selection based on the state
def choose_action(state):
    """Choose action using epsilon-greedy policy."""
    if random.uniform(0, 1) < EPSILON:  # Exploration
        return random.choice(ACTIONS)
    if state not in Q_table:
        # Initialize with small random values for consistent exploration
        Q_table[state] = [random.uniform(-0.01, 0.01) for _ in ACTIONS]
    return ACTIONS[np.argmax(Q_table[state])]


# Q-learning update - OPTIMIZED
def update_q_value(state, action, reward, next_state, is_terminal=False):
    """
    Update Q-value using Q-learning algorithm.
    Handles terminal states correctly (no future reward).
    """
    if state not in Q_table:
        Q_table[state] = [random.uniform(-0.01, 0.01) for _ in ACTIONS]

    action_index = ACTIONS.index(action)

    if is_terminal:
        # Terminal state: no future reward
        target = reward
    else:
        if next_state not in Q_table:
            Q_table[next_state] = [random.uniform(-0.01, 0.01) for _ in ACTIONS]
        max_future_q = max(Q_table[next_state])
        target = reward + GAMMA * max_future_q

    # Q-learning update
    Q_table[state][action_index] += ALPHA * (target - Q_table[state][action_index])


# Store experience in memory buffer - OPTIMIZED
def store_in_memory(state, action, reward, next_state, is_terminal=False):
    """Store experience in replay buffer (deque automatically handles max size)."""
    memory.append((state, action, reward, next_state, is_terminal))


# Sample a batch from memory and update Q-values - OPTIMIZED
def replay():
    """Sample random batch from memory and perform Q-learning updates."""
    if len(memory) < BATCH_SIZE:
        return
    batch = random.sample(memory, BATCH_SIZE)
    for state, action, reward, next_state, is_terminal in batch:
        update_q_value(state, action, reward, next_state, is_terminal)

# Main game loop - FULLY OPTIMIZED
def run_game():
    """
    Run a single game of snake with Q-learning.
    Returns: (final_score, steps_taken)
    """
    global step_counter

    snake = initial_snake.copy()
    direction = random.choice(directions)
    food = spawn_food(snake)
    score = 0
    steps = 0
    steps_since_food = 0

    # Track previous distance for reward shaping
    prev_distance = manhattan_distance(snake[0], food)

    while True:
        # Get state using new relative encoding from utils
        state = get_state(snake, food, direction)

        action = choose_action(state)

        # Move snake
        game_continue, new_snake, new_direction = move_snake(snake, direction, action, food)
        steps += 1
        steps_since_food += 1
        step_counter += 1

        # Check for termination conditions
        if not game_continue:
            # Game over - terminal state
            reward = PENALTY_DEATH
            store_in_memory(state, action, reward, state, is_terminal=True)
            update_q_value(state, action, reward, state, is_terminal=True)
            return score, steps

        # Check for infinite loop (snake not eating for too long)
        if steps_since_food > MAX_STEPS_WITHOUT_FOOD:
            # Terminate game to prevent endless wandering
            reward = PENALTY_DEATH / 2  # Less harsh penalty
            store_in_memory(state, action, reward, state, is_terminal=True)
            update_q_value(state, action, reward, state, is_terminal=True)
            return score, steps

        # Calculate reward
        if new_snake[0] == food:
            # Snake ate food!
            reward = REWARD_FOOD
            score += REWARD_FOOD
            food = spawn_food(new_snake)
            steps_since_food = 0  # Reset counter
            prev_distance = manhattan_distance(new_snake[0], food)
        else:
            # Normal move - apply distance-based reward shaping
            new_distance = manhattan_distance(new_snake[0], food)

            if new_distance < prev_distance:
                # Moving closer to food
                reward = PENALTY_MOVE + REWARD_CLOSER
            else:
                # Moving away from food
                reward = PENALTY_MOVE + PENALTY_FARTHER

            prev_distance = new_distance

        # Update state
        snake = new_snake
        direction = new_direction
        next_state = get_state(snake, food, direction)

        # Store experience
        store_in_memory(state, action, reward, next_state, is_terminal=False)

        # Periodic replay for efficiency (only every REPLAY_FREQUENCY steps)
        if step_counter % REPLAY_FREQUENCY == 0:
            replay()


# Function to compute moving average
def moving_average(data, window_size):
    """Calculate moving average for plotting."""
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


def run_multiple_games(num_games=NUM_GAMES):
    """Run multiple games and track training progress."""
    global EPSILON
    total_score = 0
    last_100_avg_score = 0
    last_100_avg_steps = 0

    for i in range(1, num_games + 1):
        # Run game and get results
        game_score, game_steps = run_game()

        # Store metrics
        scores.append(game_score)
        steps_per_game.append(game_steps)
        total_score += game_score
        cumulative_scores.append(total_score)
        iterations.append(i)
        epsilons.append(EPSILON)

        # Decay epsilon
        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

        # Update moving averages every 100 iterations
        if i % MOVING_AVERAGE_CHART_VAL == 0:
            last_100_avg_score = np.mean(scores[-MOVING_AVERAGE_CHART_VAL:])
            last_100_avg_steps = np.mean(steps_per_game[-MOVING_AVERAGE_CHART_VAL:])

        # Print progress
        sys.stdout.write(
            f"\rGame {i}/{num_games} | "
            f"Score: {game_score:.1f} | "
            f"Avg-{MOVING_AVERAGE_CHART_VAL}: {last_100_avg_score:.2f} | "
            f"Steps: {game_steps} | "
            f"Epsilon: {EPSILON:.4f} | "
            f"Q-table: {len(Q_table)} states"
        )
        sys.stdout.flush()

    print(f"\n\nTraining completed!")
    print(f"Total Score: {total_score:.2f}")
    print(f"Average Score: {total_score/num_games:.2f}")
    print(f"Final Q-table size: {len(Q_table)} states")

# Run the simulation
run_multiple_games(NUM_GAMES)

# Save the trained Q-table using pickle
print("\nSaving trained Q-table...")
with open('trained_q_table.pkl', 'wb') as f:
    pickle.dump(Q_table, f)
print("Q-table saved to 'trained_q_table.pkl'")

# Calculate the moving averages for plotting
moving_avg_scores = moving_average(scores, MOVING_AVERAGE_CHART_VAL)
moving_avg_steps = moving_average(steps_per_game, MOVING_AVERAGE_CHART_VAL)

# Align iterations with the moving averages
valid_iterations = iterations[len(iterations) - len(moving_avg_scores):]

# Plot the graphs - ENHANCED
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Plot epsilon vs iterations
ax1.plot(iterations, epsilons, color='blue', linewidth=1.5)
ax1.set_xlabel('Game Number')
ax1.set_ylabel('Epsilon (Exploration Rate)')
ax1.set_title('Epsilon Decay Over Training')
ax1.grid(True, alpha=0.3)

# Plot individual game scores vs iterations
ax2.plot(iterations, scores, color='green', alpha=0.2, label='Individual Scores')
ax2.plot(valid_iterations, moving_avg_scores, color='darkgreen', linewidth=2,
         label=f'{MOVING_AVERAGE_CHART_VAL}-Game Moving Avg')
ax2.set_xlabel('Game Number')
ax2.set_ylabel('Score')
ax2.set_title('Score per Game')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot cumulative score vs iterations
ax3.plot(iterations, cumulative_scores, color='red', linewidth=1.5)
ax3.set_xlabel('Game Number')
ax3.set_ylabel('Cumulative Score')
ax3.set_title('Cumulative Score Over Training')
ax3.grid(True, alpha=0.3)

# Plot steps per game (NEW)
ax4.plot(iterations, steps_per_game, color='purple', alpha=0.2, label='Steps per Game')
ax4.plot(valid_iterations, moving_avg_steps, color='darkviolet', linewidth=2,
         label=f'{MOVING_AVERAGE_CHART_VAL}-Game Moving Avg')
ax4.set_xlabel('Game Number')
ax4.set_ylabel('Steps')
ax4.set_title('Steps Survived per Game')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
print("\nDisplaying training visualization...")
plt.show()
