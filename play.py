import pygame
import pickle
import random
import time
import numpy as np

GRID_SIZE = 10
ACTIONS = ["straight", "left", "right"]
WIDTH, HEIGHT = GRID_SIZE * 10, GRID_SIZE * 10  # Set the screen size to 100x100 pixels

# Initialize Pygame
pygame.init()

# Setup for the game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")

# Initialize the snake and its direction
initial_snake = [(5, 5)]  # Snake starts at (5, 5)
directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Up, Right, Down, Left

# Load the trained Q-table
with open('trained_q_table.pkl', 'rb') as f:
    Q_table = pickle.load(f)

# Rewards & Penalties
REWARD_FOOD = 1.0
PENALTY_WALL = -1.0
PENALTY_TAIL = -1.0
PENALTY_MOVE = -0.05

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

# Epsilon-greedy action selection based on the state (since no exploration, it's pure exploitation)
def choose_action(state):
    if state not in Q_table:
        Q_table[state] = [0] * len(ACTIONS)  # Initialize Q-values for new state
    action = ACTIONS[np.argmax(Q_table[state])]  # Choose action with the highest Q-value
    return action

# Move the snake based on the chosen action
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
    if new_head in snake[1:]:
        return False, snake, new_direction  # Hit tail
    # Move snake: add new head and remove last segment
    if new_head == food:
        snake = [new_head] + snake  # Add new head to the snake (growing)
    else:
        snake = [new_head] + snake  # Add new head to the snake
        snake.pop()  # Remove the tail segment (snake moves forward)
    return True, snake, new_direction

# Function to spawn food at random locations
def spawn_food(snake):
    while True:
        food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        if food not in snake:
            return food

# Main loop to run the game with the trained model
def play_game():
    while True:  # Outer loop to restart the game
        snake = initial_snake.copy()
        direction = random.choice(directions)
        food = spawn_food(snake)
        score = 0
        clock = pygame.time.Clock()

        # Run the game loop
        game_continue = True
        while game_continue:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return  # Exit the program if the user closes the window

            # Get state (food and snake direction)
            food_dx, food_dy = get_deltas(snake, food)
            wall_and_tail = get_wall_and_tail_proximity(snake)
            # Create state: (food_dx, food_dy, current direction index, wall/tail proximities)
            state = (food_dx, food_dy, directions.index(direction)) + wall_and_tail
            
            # Choose action based on Q-table (no exploration)
            action = choose_action(state)

            # Move snake and check if the game is over
            game_continue, snake, direction = move_snake(snake, direction, action, food)

            # Check if snake eats food
            if snake[0] == food:
                score += 10
                food = spawn_food(snake)

            # Draw the game state
            screen.fill((0, 0, 0))  # Clear screen
            for segment in snake:
                pygame.draw.rect(screen, (0, 255, 0), (segment[0] * 10, segment[1] * 10, 10, 10))
            pygame.draw.rect(screen, (255, 0, 0), (food[0] * 10, food[1] * 10, 10, 10))

            # Update the screen and set the FPS
            pygame.display.flip()
            clock.tick(10)

        print(f"Game Over! Final Score: {score}")
        time.sleep(2)  # Pause for 2 seconds before restarting

# Run the game with the trained model
play_game()
