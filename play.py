import pygame
import pickle
import random

# Constants for the game
GRID_SIZE = 10
ACTIONS = ["straight", "left", "right"]
WIDTH, HEIGHT = 100, 100  # Set the screen size to 100x100 pixels

# Initialize Pygame
pygame.init()

# Setup for the game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")

# Initialize the snake and its direction
initial_snake = [(0, 0)]  # Snake starts at (0, 0)
directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Up, Right, Down, Left
direction = random.choice(directions)

# Load the trained Q-table
with open('trained_q_table.pkl', 'rb') as f:
    Q_table = pickle.load(f)

# Function to get the dx/dy to food and the tail (used for decision-making)
def get_deltas(snake, food):
    head = snake[0]
    food_dx = food[0] - head[0]
    food_dy = food[1] - head[1]
    tail_deltas = []
    for segment in snake[1:]:
        tail_dx = head[0] - segment[0]
        tail_dy = head[1] - segment[1]
        tail_deltas.append((tail_dx, tail_dy))
    return (food_dx, food_dy), tail_deltas

# Function to choose action based on the trained Q-table (exploitation)
def choose_action(state):
    if state not in Q_table:
        Q_table[state] = [0] * len(ACTIONS)  # Initialize Q-values for new state
    action = ACTIONS[max(range(len(Q_table[state])), key=lambda x: Q_table[state][x])]
    return action

# Function to move the snake and render the display
def move_snake(snake, direction, action, food):
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

    # Snake grows by adding new head if it eats food
    if new_head == food:
        snake = [new_head] + snake  # Add new head to the snake
    else:
        snake = [new_head] + snake  # Add new head to the snake
        snake.pop()  # Remove the tail segment (snake moves forward)

    return True, snake, new_direction


# Function to spawn food at random locations
def spawn_food(snake):
    while True:
        food = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
        if food not in snake:
            return food

# Main loop to run the game with the trained model
def play_game():
    snake = initial_snake.copy()
    direction = random.choice(directions)  # Initialize direction randomly
    food = spawn_food(snake)
    score = 0
    clock = pygame.time.Clock()

    # Run the game loop
    game_continue = True
    while game_continue:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_continue = False

        # Get state (food and snake direction)
        (food_dx, food_dy), tail_deltas = get_deltas(snake, food)
        state = (food_dx, food_dy, direction)

        # Choose action based on Q-table
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
    pygame.quit()

# Run the game with the trained model
play_game()