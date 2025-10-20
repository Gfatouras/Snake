"""
Play the Snake game using a trained Q-learning model.
Displays the game using Pygame with the trained agent playing.
"""
import pygame
import pickle
import random
import time
import numpy as np
from utils import (
    GRID_SIZE, ACTIONS, directions, get_state,
    spawn_food, move_snake
)

# Display settings
CELL_SIZE = 20  # Increased for better visibility
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE
FPS = 10  # Game speed

# Initialize Pygame
pygame.init()

# Setup for the game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game - Q-Learning Agent")

# Colors
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 180, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

# Initialize the snake and its direction
initial_snake = [(5, 5)]  # Snake starts at (5, 5)

# Load the trained Q-table
print("Loading trained Q-table...")
with open('trained_q_table.pkl', 'rb') as f:
    Q_table = pickle.load(f)
print(f"Q-table loaded with {len(Q_table)} states")


def choose_action(state):
    """Choose action using pure exploitation (no exploration)."""
    if state not in Q_table:
        # If state not in Q-table, choose random action
        return random.choice(ACTIONS)
    # Choose action with highest Q-value
    return ACTIONS[np.argmax(Q_table[state])]

# Main loop to run the game with the trained model
def play_game():
    """Main game loop displaying the trained agent playing Snake."""
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    while True:  # Outer loop to restart the game
        snake = initial_snake.copy()
        direction = random.choice(directions)
        food = spawn_food(snake)
        score = 0
        steps = 0
        clock = pygame.time.Clock()

        # Run the game loop
        game_continue = True
        while game_continue:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return  # Exit the program if the user closes the window
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # Pause game on spacebar
                        paused = True
                        while paused:
                            for e in pygame.event.get():
                                if e.type == pygame.QUIT:
                                    pygame.quit()
                                    return
                                if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                                    paused = False

            # Get state using new relative encoding
            state = get_state(snake, food, direction)

            # Choose action based on Q-table (no exploration)
            action = choose_action(state)

            # Move snake and check if the game is over
            game_continue, snake, direction = move_snake(snake, direction, action, food)
            steps += 1

            # Check if snake eats food
            if game_continue and snake[0] == food:
                score += 1  # Count number of food eaten
                food = spawn_food(snake)

            # Draw the game state
            screen.fill(BLACK)

            # Draw grid (optional - for better visibility)
            for x in range(0, WIDTH, CELL_SIZE):
                pygame.draw.line(screen, (40, 40, 40), (x, 0), (x, HEIGHT))
            for y in range(0, HEIGHT, CELL_SIZE):
                pygame.draw.line(screen, (40, 40, 40), (0, y), (WIDTH, y))

            # Draw snake with gradient effect
            for i, segment in enumerate(snake):
                if i == 0:
                    # Head - brighter green
                    color = GREEN
                else:
                    # Body - darker green
                    color = DARK_GREEN
                pygame.draw.rect(screen, color,
                               (segment[0] * CELL_SIZE + 1, segment[1] * CELL_SIZE + 1,
                                CELL_SIZE - 2, CELL_SIZE - 2))

            # Draw food
            pygame.draw.rect(screen, RED,
                           (food[0] * CELL_SIZE + 1, food[1] * CELL_SIZE + 1,
                            CELL_SIZE - 2, CELL_SIZE - 2))

            # Draw score and info
            score_text = font.render(f'Score: {score}', True, WHITE)
            length_text = small_font.render(f'Length: {len(snake)}', True, WHITE)
            steps_text = small_font.render(f'Steps: {steps}', True, WHITE)

            screen.blit(score_text, (10, 10))
            screen.blit(length_text, (10, 50))
            screen.blit(steps_text, (10, 75))

            # Update the screen and set the FPS
            pygame.display.flip()
            clock.tick(FPS)

        # Game over screen
        screen.fill(BLACK)
        game_over_text = font.render('Game Over!', True, RED)
        final_score_text = font.render(f'Final Score: {score}', True, WHITE)
        final_length_text = small_font.render(f'Final Length: {len(snake)}', True, WHITE)
        restart_text = small_font.render('Restarting in 3 seconds...', True, WHITE)

        screen.blit(game_over_text, (WIDTH//2 - 100, HEIGHT//2 - 60))
        screen.blit(final_score_text, (WIDTH//2 - 120, HEIGHT//2 - 10))
        screen.blit(final_length_text, (WIDTH//2 - 100, HEIGHT//2 + 20))
        screen.blit(restart_text, (WIDTH//2 - 140, HEIGHT//2 + 60))
        pygame.display.flip()

        print(f"Game Over! Final Score: {score} | Length: {len(snake)} | Steps: {steps}")
        time.sleep(3)  # Pause for 3 seconds before restarting


# Run the game with the trained model
if __name__ == "__main__":
    print("Starting Snake game with trained Q-learning agent...")
    print("Press SPACE to pause/unpause")
    print("Close window to exit")
    play_game()
