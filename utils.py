"""
Shared utilities for Snake Q-learning agent.
Contains common functions used by both training (main.py) and playing (play.py).
"""
import numpy as np
import random

# Constants for the game
GRID_SIZE = 10
ACTIONS = ["straight", "left", "right"]
directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Up, Right, Down, Left


def get_deltas(snake, food):
    """
    Get the normalized difference between snake's head and food.
    Returns values in {-1, 0, 1} for each axis.
    """
    head = snake[0]
    food_dx = np.sign(food[0] - head[0])
    food_dy = np.sign(food[1] - head[1])
    return food_dx, food_dy


def get_danger_proximity(snake, direction):
    """
    Get danger (wall or tail) in three directions relative to current heading:
    - straight ahead
    - to the left
    - to the right

    This creates a direction-invariant state space that's 4x smaller.
    Returns: (danger_straight, danger_left, danger_right) where 1 = danger, 0 = safe
    """
    head_x, head_y = snake[0]
    current_dir_idx = directions.index(direction)

    # Get the three directions relative to current heading
    straight_dir = directions[current_dir_idx]
    left_dir = directions[(current_dir_idx - 1) % 4]
    right_dir = directions[(current_dir_idx + 1) % 4]

    def is_danger(pos):
        """Check if a position is dangerous (wall or tail)"""
        x, y = pos
        # Check wall collision
        if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
            return 1
        # Check tail collision
        if pos in snake:
            return 1
        return 0

    # Check positions in each direction
    straight_pos = (head_x + straight_dir[0], head_y + straight_dir[1])
    left_pos = (head_x + left_dir[0], head_y + left_dir[1])
    right_pos = (head_x + right_dir[0], head_y + right_dir[1])

    return (is_danger(straight_pos), is_danger(left_pos), is_danger(right_pos))


def get_relative_food_direction(snake, food, direction):
    """
    Get food direction relative to snake's current heading.
    Returns: (food_vertical, food_horizontal) where each is in {-1, 0, 1}

    - food_vertical: -1 = behind, 0 = aligned, 1 = ahead
    - food_horizontal: -1 = left, 0 = aligned, 1 = right

    This creates a cleaner 2D relative representation that's direction-invariant.
    """
    head = snake[0]
    current_dir_idx = directions.index(direction)

    # Get actual food direction
    food_dx = np.sign(food[0] - head[0])
    food_dy = np.sign(food[1] - head[1])

    # Get the forward and right vectors for current direction
    forward_dir = directions[current_dir_idx]
    right_dir = directions[(current_dir_idx + 1) % 4]

    # Project food vector onto forward (vertical) and right (horizontal) axes
    food_vertical = food_dx * forward_dir[0] + food_dy * forward_dir[1]
    food_horizontal = food_dx * right_dir[0] + food_dy * right_dir[1]

    return (int(np.sign(food_vertical)), int(np.sign(food_horizontal)))


def get_state(snake, food, direction):
    """
    Generate state representation using relative encoding.
    State space: 3^2 (food) × 2^3 (danger) = 9 × 8 = 72 possible states

    This is direction-invariant (4x smaller than absolute encoding) and
    uses a clean 2D relative coordinate system.
    """
    food_rel = get_relative_food_direction(snake, food, direction)
    danger = get_danger_proximity(snake, direction)
    # State: (food_vertical, food_horizontal, danger_straight, danger_left, danger_right)
    return food_rel + danger


def spawn_food(snake):
    """
    Spawn food at random location ensuring it doesn't overlap with snake.
    """
    while True:
        food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        if food not in snake:
            return food


def move_snake(snake, direction, action, food):
    """
    Move the snake based on the chosen action.
    Returns: (game_continue, new_snake, new_direction)
    """
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

    # If the new head is on the food, grow the snake
    if new_head == food:
        new_snake = [new_head] + snake  # Grow snake by adding new head without removing tail
    else:
        new_snake = [new_head] + snake[:-1]  # Normal movement: add new head, remove tail

    return True, new_snake, new_direction


def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
