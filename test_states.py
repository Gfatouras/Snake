"""Test to see how many unique states we can generate."""
import random
from utils import get_state, directions, spawn_food

# Generate lots of random game scenarios
unique_states = set()

for _ in range(10000):
    # Random snake position and length
    snake_length = random.randint(1, 20)
    snake = [(random.randint(0, 9), random.randint(0, 9))]

    # Add body segments
    for i in range(snake_length - 1):
        last = snake[-1]
        # Try to add adjacent segment
        possible = [(last[0]+dx, last[1]+dy) for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]]
        valid = [p for p in possible if 0 <= p[0] < 10 and 0 <= p[1] < 10 and p not in snake]
        if valid:
            snake.append(random.choice(valid))

    # Random direction
    direction = random.choice(directions)

    # Random food
    food = spawn_food(snake)

    # Get state
    state = get_state(snake, food, direction)
    unique_states.add(state)

print(f"Unique states found: {len(unique_states)}")
print(f"Theoretical maximum: 3 * 3 * 3 * 2 * 2 * 2 = 216")
print(f"\nSample states:")
for i, state in enumerate(list(unique_states)[:10]):
    print(f"  {state}")

# Check if there's a pattern
print(f"\nAnalyzing state components:")
food_straights = set()
food_lefts = set()
food_rights = set()
danger_straights = set()
danger_lefts = set()
danger_rights = set()

for state in unique_states:
    food_straights.add(state[0])
    food_lefts.add(state[1])
    food_rights.add(state[2])
    danger_straights.add(state[3])
    danger_lefts.add(state[4])
    danger_rights.add(state[5])

print(f"Food straight values: {sorted(food_straights)}")
print(f"Food left values: {sorted(food_lefts)}")
print(f"Food right values: {sorted(food_rights)}")
print(f"Danger straight values: {sorted(danger_straights)}")
print(f"Danger left values: {sorted(danger_lefts)}")
print(f"Danger right values: {sorted(danger_rights)}")
