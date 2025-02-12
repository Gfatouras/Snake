import random
import pickle  # To save the model
import matplotlib.pyplot as plt

# Constants for the game
GRID_SIZE = 10
ACTIONS = ["straight", "left", "right"]
ALPHA = 0.1  # Learning rate
GAMMA = 0.90  # Discount factor
EPSILON = 1.0  # Initial exploration rate
EPSILON_DECAY = 0.9997  # Slower decay
EPSILON_MIN = 0.0  # Minimum exploration rate
MEMORY_SIZE = 100  # Memory buffer size
BATCH_SIZE = 64  # Batch size for training

# Snake parameters
initial_snake = [(0, 0)]  # Snake starts at (0, 0)
directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Up, Right, Down, Left

# Initialize Q-table
Q_table = {}

# Memory buffer
memory = []

# Function to get the dx/dy to food and the tail (used for decision-making)
def get_deltas(snake, food):
    head = snake[0]
    food_dx = food[0] - head[0]
    food_dy = food[1] - head[1]

    # dx/dy to tail
    tail_deltas = []
    for segment in snake[1:]:
        tail_dx = head[0] - segment[0]
        tail_dy = head[1] - segment[1]
        tail_deltas.append((tail_dx, tail_dy))
    
    return (food_dx, food_dy), tail_deltas

# Function to choose action with epsilon-greedy strategy
def choose_action(state, direction):
    if random.uniform(0, 1) < EPSILON:  # Exploration
        action = random.choice(ACTIONS)
    else:  # Exploitation
        if state not in Q_table:
            Q_table[state] = [0] * len(ACTIONS)  # Initialize Q-values for new state
        action = ACTIONS[max(range(len(Q_table[state])), key=lambda x: Q_table[state][x])]
    return action

# Function to move the snake based on action
def move_snake(snake, direction, action):
    head = snake[0]
    
    # If action is "straight", keep the current direction
    if action == "straight":
        new_direction = direction
    elif action == "left":
        # Turn left (counterclockwise 90 degrees)
        new_direction = directions[(directions.index(direction) - 1) % 4]
    elif action == "right":
        # Turn right (clockwise 90 degrees)
        new_direction = directions[(directions.index(direction) + 1) % 4]

    # Calculate new head position based on new direction
    new_head = (head[0] + new_direction[0], head[1] + new_direction[1])

    # Check if the snake goes out of bounds
    if new_head[0] < 0 or new_head[0] >= GRID_SIZE or new_head[1] < 0 or new_head[1] >= GRID_SIZE:
        return False, snake, new_direction  # Game over (hit the wall)

    # Check if the snake runs into itself
    if new_head in snake[1:]:
        return False, snake, new_direction  # Game over (hit its own tail)

    # Snake grows by adding new head
    snake = [new_head] + snake

    return True, snake, new_direction


# Function to spawn new food
def spawn_food(snake):
    while True:
        food = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
        if food not in snake:
            return food

# Function to update Q-values using the Q-learning formula
def update_q_value(state, action, reward, next_state):
    if state not in Q_table:
        Q_table[state] = [0] * len(ACTIONS)
    if next_state not in Q_table:
        Q_table[next_state] = [0] * len(ACTIONS)

    # Get the maximum Q-value for the next state
    max_future_q = max(Q_table[next_state])

    # Update Q-value using Q-learning formula
    action_index = ACTIONS.index(action)
    Q_table[state][action_index] += ALPHA * (reward + GAMMA * max_future_q - Q_table[state][action_index])

# Function to store experiences in memory buffer
def store_in_memory(state, action, reward, next_state):
    if len(memory) >= MEMORY_SIZE:
        memory.pop(0)  # Remove the oldest memory if the buffer is full
    memory.append((state, action, reward, next_state))

# Function to sample a batch from memory and update Q-values
def replay():
    if len(memory) < BATCH_SIZE:
        return

    batch = random.sample(memory, BATCH_SIZE)
    for state, action, reward, next_state in batch:
        update_q_value(state, action, reward, next_state)

# Function to run a game simulation
def run_game():
    snake = initial_snake.copy()
    direction = random.choice(directions)  # Random initial direction
    food = spawn_food(snake)  # Initialize food position
    score = 0

    # Run until game over
    while True:
        # Get deltas (dx/dy) to food and tail (tail is used for avoiding collisions)
        (food_dx, food_dy), tail_deltas = get_deltas(snake, food)

        # Represent the state
        state = (food_dx, food_dy, direction)

        # Choose action based on epsilon-greedy strategy
        action = choose_action(state, direction)

        # Move snake based on chosen action
        game_continue, snake, direction = move_snake(snake, direction, action)

        if not game_continue:
            reward = -1  # If the snake crashes, return a penalty
            store_in_memory(state, action, reward, state)  # Store the experience
            replay()  # Sample from memory and update Q-table
            return score + reward

        # Check if snake eats food
        if snake[0] == food:
            reward = 1  # Reward for eating food
            score += reward
            food = spawn_food(snake)  # Spawn new food
        else:
            reward = -0.1  # No reward, snake just moves forward
            snake.pop()  # Remove the tail segment (snake moves forward)

        # Get the next state after the move
        next_state = (food_dx, food_dy, direction)

        # Store experience in memory
        store_in_memory(state, action, reward, next_state)

        # Sample from memory and update Q-table
        replay()

        # Limit the number of steps per game for this simulation
        if len(snake) > 50:  # Game limit (snake growing too long)
            break

    return score

# Track epsilon, score per game, cumulative score, and episode number
epsilons = []
scores = []  # Individual game scores
cumulative_scores = []  # Cumulative sum of scores over time
iterations = []

# Function to run multiple games and track epsilon, score, and iteration
def run_multiple_games(num_games=20000):
    global EPSILON  # Declare EPSILON as global before modifying it
    total_score = 0
    for i in range(num_games):
        game_score = run_game()
        scores.append(game_score)  # Store individual game score
        total_score += game_score  # Update cumulative score
        cumulative_scores.append(total_score)  # Store cumulative score
        iterations.append(i + 1)
        epsilons.append(EPSILON)

        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

    print(f"Total Score after {num_games} games: {total_score}")

# Run the simulation for 10000 games
run_multiple_games(20000)

# Save the trained Q-table using pickle
with open('trained_q_table.pkl', 'wb') as f:
    pickle.dump(Q_table, f)

# Plot the graphs for epsilon, score per game, cumulative score, and iterations
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

# Plot epsilon vs iterations
ax1.plot(iterations, epsilons, color='blue')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Epsilon')
ax1.set_title('Epsilon vs Iterations')

# Plot individual game scores vs iterations
ax2.plot(iterations, scores, color='green')
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Score')
ax2.set_title('Score per Game')

# Plot cumulative score vs iterations
ax3.plot(iterations, cumulative_scores, color='red')
ax3.set_xlabel('Iterations')
ax3.set_ylabel('Cumulative Score')
ax3.set_title('Cumulative Score vs Iterations')

plt.tight_layout()
plt.show()


