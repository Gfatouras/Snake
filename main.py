import random
import pickle  # To save the model
import matplotlib.pyplot as plt
import sys
import numpy as np  # For calculating moving average

# Constants for the game
GRID_SIZE = 10
ACTIONS = ["straight", "left", "right"]

ALPHA = 0.1      # Learning rate
GAMMA = 0.90     # Discount factor
EPSILON = 1.0    # Initial exploration rate
EPSILON_DECAY = 0.9993  # Decay rate
EPSILON_MIN = 0.0       # Minimum exploration rate

MEMORY_SIZE = 1024  # Memory buffer size
BATCH_SIZE = 128    # Batch size for training
NUM_GAMES = 5_000   # Number of iterations

MOVING_AVERAGE_CHART_VAL = 100  # Last 100 games average

# Rewards & Penalties
REWARD_FOOD = 1.0       # Reward for eating food
PENALTY_WALL = -1.0     # Penalty for hitting a wall
PENALTY_TAIL = -1.0     # Penalty for hitting own tail
PENALTY_MOVE = -0.05    # Small penalty for each movement to encourage efficiency

# Snake parameters
initial_snake = [(5, 5)]  # Snake starts at the center of the grid
directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Up, Right, Down, Left

# Initialize Q-table and Memory buffer
Q_table = {}
memory = []

# Lists for plotting
epsilons = []
scores = []            # Individual game scores
cumulative_scores = [] # Cumulative sum of scores over time
iterations = []

# Function to get the normalized difference between snake's head and food
def get_deltas(snake, food):
    head = snake[0]
    food_dx = np.sign(food[0] - head[0])  # Normalize to -1, 0, or 1
    food_dy = np.sign(food[1] - head[1])
    return food_dx, food_dy

# Function to detect proximity of walls and tail around the snake's head
def get_wall_and_tail_proximity(snake):
    head_x, head_y = snake[0]
    # Walls: 1 if the head is at the boundary, else 0
    wall_left = 1 if head_x == 0 else 0
    wall_right = 1 if head_x == GRID_SIZE - 1 else 0
    wall_up = 1 if head_y == 0 else 0
    wall_down = 1 if head_y == GRID_SIZE - 1 else 0
    # Tail proximity: check immediate adjacent positions
    left_pos = (head_x - 1, head_y)
    right_pos = (head_x + 1, head_y)
    up_pos = (head_x, head_y - 1)
    down_pos = (head_x, head_y + 1)
    tail_left = 1 if left_pos in snake else 0
    tail_right = 1 if right_pos in snake else 0
    tail_up = 1 if up_pos in snake else 0
    tail_down = 1 if down_pos in snake else 0
    return wall_left, wall_right, wall_up, wall_down, tail_left, tail_right, tail_up, tail_down

# Epsilon-greedy action selection based on the state
def choose_action(state):
    global EPSILON
    if random.uniform(0, 1) < EPSILON:  # Exploration
        return random.choice(ACTIONS)
    if state not in Q_table:
        Q_table[state] = [random.uniform(-1, 1) for _ in ACTIONS]
    return ACTIONS[np.argmax(Q_table[state])]

# Move the snake based on the chosen action.
# Now it takes the food position as a parameter so we can grow the snake.
def move_snake(snake, direction, action, food):
    head = snake[0]
    new_direction = direction
    if action == "left":
        new_direction = directions[(directions.index(direction) - 1) % 4]
    elif action == "right":
        new_direction = directions[(directions.index(direction) + 1) % 4]
    new_head = (head[0] + new_direction[0], head[1] + new_direction[1])
    
    # Check for wall collision
    if new_head[0] < 0 or new_head[0] >= GRID_SIZE or new_head[1] < 0 or new_head[1] >= GRID_SIZE:
        return False, snake, new_direction  # Hit wall
    
    # Check for collision with itself
    if new_head in snake:
        return False, snake, new_direction  # Hit tail
    
    # If the new head is on the food, grow the snake (do not remove the tail)
    if new_head == food:
        new_snake = [new_head] + snake  # Grow snake by adding new head without removing tail
    else:
        new_snake = [new_head] + snake[:-1]  # Normal movement: add new head, remove tail
    
    return True, new_snake, new_direction

# Q-learning update
def update_q_value(state, action, reward, next_state):
    if state not in Q_table:
        Q_table[state] = [random.uniform(-0.1, 0.1) for _ in ACTIONS]
    if next_state not in Q_table:
        Q_table[next_state] = [random.uniform(-0.1, 0.1) for _ in ACTIONS]
    action_index = ACTIONS.index(action)
    max_future_q = max(Q_table[next_state])
    Q_table[state][action_index] += ALPHA * (reward + GAMMA * max_future_q - Q_table[state][action_index])

# Store experience in memory buffer
def store_in_memory(state, action, reward, next_state):
    if len(memory) >= MEMORY_SIZE:
        memory.pop(0)
    memory.append((state, action, reward, next_state))

# Sample a batch from memory and update Q-values
def replay():
    if len(memory) < BATCH_SIZE:
        return
    batch = random.sample(memory, BATCH_SIZE)
    for state, action, reward, next_state in batch:
        update_q_value(state, action, reward, next_state)

# Function to spawn food at random locations, ensuring it doesn't spawn on the snake's body (head or tail)
def spawn_food(snake):
    while True:
        food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        if food not in snake:  # Ensure food doesn't spawn on the snake's body
            return food

# Main game loop
def run_game():
    snake = initial_snake.copy()
    direction = random.choice(directions)
    food = spawn_food(snake)
    score = 0
    while True:
        food_dx, food_dy = get_deltas(snake, food)
        wall_and_tail = get_wall_and_tail_proximity(snake)
        # Create state: (food_dx, food_dy, current direction index, wall/tail proximities)
        state = (food_dx, food_dy, directions.index(direction)) + wall_and_tail
        
        action = choose_action(state)
        # Pass the food position into move_snake so that growth is handled here
        game_continue, snake, direction = move_snake(snake, direction, action, food)
        
        # If game over, apply penalty based on whether it hit a wall or its tail
        if not game_continue:
            # If the head is in the rest of the body, it's a tail collision.
            reward = PENALTY_TAIL if snake[0] in snake[1:] else PENALTY_WALL
            update_q_value(state, action, reward, state)
            return score + reward

        # Check if snake has eaten the food
        if snake[0] == food:
            reward = REWARD_FOOD
            score += reward
            food = spawn_food(snake)  # Spawn new food ensuring it does not appear on the snake
        else:
            reward = PENALTY_MOVE

        next_food_dx, next_food_dy = get_deltas(snake, food)
        next_wall_and_tail = get_wall_and_tail_proximity(snake)
        next_state = (next_food_dx, next_food_dy, directions.index(direction)) + next_wall_and_tail

        update_q_value(state, action, reward, next_state)
        store_in_memory(state, action, reward, next_state)
        replay()
        
        # Optional termination condition (for very long games)
        if len(snake) > 50:
            break
    return score

# Function to compute moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def run_multiple_games(num_games=NUM_GAMES):
    global EPSILON  # Declare EPSILON as global before modifying it
    total_score = 0
    last_100_avg = 0  # Store last computed moving average

    for i in range(1, num_games + 1):  # Start from 1 to avoid division by zero
        game_score = run_game()
        scores.append(game_score)  # Store individual game score
        total_score += game_score  # Update cumulative score
        cumulative_scores.append(total_score)  # Store cumulative score
        iterations.append(i)
        epsilons.append(EPSILON)

        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

        # Update last_100_avg every 100 iterations
        if i % MOVING_AVERAGE_CHART_VAL == 0:
            last_100_avg = np.mean(scores[-MOVING_AVERAGE_CHART_VAL:])
        
        # Print progress
        sys.stdout.write(f"\rGame {i}/{num_games}: Total Score = {total_score}, "
                         f"Last {MOVING_AVERAGE_CHART_VAL} Avg Score = {last_100_avg:.2f}, "
                         f"Epsilon = {EPSILON:.4f}, Q-table size = {len(Q_table)}")
        sys.stdout.flush()  # Ensure output is updated immediately

    print(f"\nTotal Score after {num_games} games: {total_score}")
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

# Run the simulation
run_multiple_games(NUM_GAMES)

# Save the trained Q-table using pickle
with open('trained_q_table.pkl', 'wb') as f:
    pickle.dump(Q_table, f)

# Calculate the moving average for plotting
moving_avg = moving_average(scores, MOVING_AVERAGE_CHART_VAL)
# Align iterations with the moving average
valid_iterations = iterations[len(iterations) - len(moving_avg):]

# Plot the graphs for epsilon, individual game scores, and cumulative score
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

# Plot epsilon vs iterations
ax1.plot(iterations, epsilons, color='blue')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Epsilon')
ax1.set_title('Epsilon vs Iterations')

# Plot individual game scores vs iterations
ax2.plot(iterations, scores, color='green', alpha=0.3, label='Individual Scores')
ax2.plot(valid_iterations, moving_avg, color='green', linestyle=':', linewidth=2, 
         label=f'{MOVING_AVERAGE_CHART_VAL} Game Moving Avg')
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Score')
ax2.set_title('Score per Game')
ax2.legend()

# Plot cumulative score vs iterations
ax3.plot(iterations, cumulative_scores, color='red')
ax3.set_xlabel('Iterations')
ax3.set_ylabel('Cumulative Score')
ax3.set_title('Cumulative Score vs Iterations')

plt.tight_layout()
plt.show()
