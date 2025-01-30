import pygame
import random
import numpy as np
import matplotlib.pyplot as plt
import threading  # For multithreading
import random
import csv

# Initialize Pygame
pygame.init()

# Constants for the game
WIDTH, HEIGHT = 200, 200
GRID_SIZE = 10
CELL_SIZE = WIDTH // GRID_SIZE

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Initialize screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")

# Q-learning parameters
alpha = 0.75  # Learning rate
gamma = 0.5  # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_decay = 0.9999  # Decay rate for epsilon
epsilon_min = 0.01  # Minimum exploration rate
Q_table = {}  # Q-table for state-action pairs

# Counters for crashes into wall vs. tail
wall_crashes = 0
tail_crashes = 0

# Function to generate the state
def get_state(snake, food):
    head = snake[0]
    food_dx = food[0] - head[0]  # Relative x position of food
    food_dy = food[1] - head[1]  # Relative y position of food
    distance_to_food = abs(food_dx) + abs(food_dy)  # Manhattan distance to food

    # Check for obstacles in each direction
    obstacle_up = (head[0], head[1] - 1) in snake or head[1] - 1 < 0
    obstacle_down = (head[0], head[1] + 1) in snake or head[1] + 1 >= GRID_SIZE
    obstacle_left = (head[0] - 1, head[1]) in snake or head[0] - 1 < 0
    obstacle_right = (head[0] + 1, head[1]) in snake or head[0] + 1 >= GRID_SIZE

    # Calculate distance to the nearest wall (up, down, left, right)
    distance_up = head[1]  # Distance from head to the top wall
    distance_down = GRID_SIZE - 1 - head[1]  # Distance from head to the bottom wall
    distance_left = head[0]  # Distance from head to the left wall
    distance_right = GRID_SIZE - 1 - head[0]  # Distance from head to the right wall
    
    # Find the minimum distance to any wall
    min_distance_to_wall = min(distance_up, distance_down, distance_left, distance_right)

    state = (
        food_dx, food_dy, distance_to_food,  # Relative food position and distance
        obstacle_up, obstacle_down, obstacle_left, obstacle_right,  # Obstacles in each direction
        len(snake),  # Snake length
        min_distance_to_wall  # Distance to the nearest wall
    )
    return state


# Function to initialize game
def reset_game():
    # Snake starts at the center of the grid
    snake = [(GRID_SIZE // 2, GRID_SIZE // 2)]  # Snake starts at the center
    food = spawn_food(snake)
    return snake, food


# Function to spawn food
def spawn_food(snake):
    while True:
        food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        if food not in snake:
            return food

# Function to update Q-value using Q-learning
def update_q_value(state, action, reward, next_state):
    if state not in Q_table:
        Q_table[state] = np.zeros(4)  # Initialize with 4 actions
    if next_state not in Q_table:
        Q_table[next_state] = np.zeros(4)
    
    max_future_q = np.max(Q_table[next_state])  # Max future Q-value
    current_q = Q_table[state][action]  # Current Q-value
    
    # Update Q-value using Q-learning formula
    Q_table[state][action] = current_q + alpha * (reward + gamma * max_future_q - current_q)

# Function to choose action (epsilon-greedy)
def choose_action(state, current_direction):
    if random.uniform(0, 1) < epsilon:  # Exploration
        # If moving horizontally (left or right), allow up/down moves
        if current_direction in [(1, 0), (-1, 0)]:  
            return random.choice([0, 2])  # Up or Down
        # If moving vertically (up or down), allow left/right moves
        else:
            return random.choice([1, 3])  # Left or Right
    else:  # Exploitation
        if state in Q_table:
            # Choose the action with the highest Q-value respecting movement constraints
            if current_direction in [(1, 0), (-1, 0)]:  # If moving horizontally
                return np.argmax([Q_table[state][0], Q_table[state][2]])  # Only up/down actions
            else:  # If moving vertically
                return np.argmax([Q_table[state][1], Q_table[state][3]])  # Only left/right actions
        return random.randint(0, 3)  # Default to random action if state is unknown

# Function to move snake
def move_snake(snake, direction):
    head = snake[0]
    new_head = (head[0] + direction[0], head[1] + direction[1])
    
    # Check if the snake collides with the walls
    if new_head[0] < 0 or new_head[0] >= GRID_SIZE or new_head[1] < 0 or new_head[1] >= GRID_SIZE:
        return False, snake  # Game over
    
    # Skip self-collision check if the snake is size 1
    if len(snake) > 1 and new_head in snake[1:]:
        return False, snake  # Death by tail
    
    snake = [new_head] + snake  # Add the new head to the front of the snake
    return True, snake


# Function to draw the game
def draw_game(snake, food, score):
    screen.fill(BLACK)
    
    # Draw the snake
    for segment in snake:
        pygame.draw.rect(screen, GREEN, (segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
    # Draw the food
    pygame.draw.rect(screen, RED, (food[0] * CELL_SIZE, food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Draw score
    font = pygame.font.SysFont(None, 24)
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (5, 5))

    pygame.display.update()

def run_episode():
    global epsilon, wall_crashes, tail_crashes
    snake, food = reset_game()
    total_reward = 0
    steps = 0
    ate_food = False
    current_direction = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])  # Start with random direction

    while True:
        state = get_state(snake, food)
        action = choose_action(state, current_direction)
        
        # Determine new direction based on the action and current direction
        if action == 0:  # Up
            new_direction = (0, -1)
        elif action == 1:  # Right
            new_direction = (1, 0)
        elif action == 2:  # Down
            new_direction = (0, 1)
        else:  # Left
            new_direction = (-1, 0)
        
        # Move the snake
        game_continue, snake = move_snake(snake, new_direction)
        
        # Check if the game is over
        if not game_continue:
            reward = -10  # Penalty for crashing into the wall
            if snake[0][0] < 0 or snake[0][0] >= GRID_SIZE or snake[0][1] < 0 or snake[0][1] >= GRID_SIZE:
                wall_crashes += 1
                cause_of_death = "Wall"  # Wall collision
            elif len(snake) >= 5 and snake[0] in snake[1:]:  # Tail collision only if snake size >= 5
                tail_crashes += 1
                cause_of_death = "Tail"
            else:
                cause_of_death = "Wall"  # For size 1, it should be wall collision, not "None"
            update_q_value(state, action, reward, state)
            print(f"Episode ended. Cause of death: {cause_of_death}. Snake size: {len(snake)} blocks. Ate food: {ate_food}")
            return total_reward, len(snake), epsilon  # End game
        
        # Check if snake eats food
        head = snake[0]
        if head == food:
            food = spawn_food(snake)
            reward = 10  # Reward for eating food
            ate_food = True
            # Snake grows after eating food, but only the new head is added, tail remains behind
        else:
            reward = -0.01 # Small penalty for each step
            snake.pop()  # Remove the tail

        # Get the next state
        next_state = get_state(snake, food)
        
        # Update Q-values
        update_q_value(state, action, reward, next_state)
        
        total_reward += reward
        steps += 1
        current_direction = new_direction  # Update the direction
        
        # Draw the game
        draw_game(snake, food, total_reward)
        pygame.time.Clock().tick(2000)  # Control FPS
        
        if steps > 1000:  # Limit the number of steps per episode
            break

    return total_reward, len(snake), epsilon

# Function for running game logic
def game_thread():
    global epsilon
    rewards_history = []
    snake_sizes = []
    epsilons = []

    for episode in range(10000):  # Train for 2000 episodes
        total_reward, snake_size, current_epsilon = run_episode()
        rewards_history.append(total_reward)
        snake_sizes.append(snake_size)
        epsilons.append(current_epsilon)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)  # Decay epsilon
        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon}")

    # Export the data to CSV after all episodes finish
    export_to_csv(rewards_history, snake_sizes, epsilons)

# Function to export data to CSV
def export_to_csv(rewards_history, snake_sizes, epsilons):
    # Define the header
    header = ["Episode", "Snake Size", "Epsilon", "Total Reward"]
    
    # Open the CSV file and write the data
    with open('episode_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for i in range(len(rewards_history)):
            writer.writerow([i + 1, snake_sizes[i], epsilons[i], rewards_history[i]])
    
    print("Episode data exported to episode_data.csv")

# Main function to run the game loop
def main():
    # Start the game logic in a separate thread
    threading.Thread(target=game_thread, daemon=True).start()

    # Main Pygame event loop (renders the screen)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        pygame.display.update()
    
    pygame.quit()

# Run the main function
if __name__ == "__main__":
    main()
