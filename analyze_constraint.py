"""Analyze the constraint in food direction encoding."""
from utils import get_relative_food_direction, directions
import numpy as np

# Test all possible food directions
print("Testing all possible (food_dx, food_dy) combinations:")
print("=" * 70)

all_combos = set()
for food_dx in [-1, 0, 1]:
    for food_dy in [-1, 0, 1]:
        # Test with each direction (should give same relative result)
        for direction in directions:
            snake = [(5, 5)]  # Center
            food = (5 + food_dx, 5 + food_dy)

            result = get_relative_food_direction(snake, food, direction)
            all_combos.add(result)

print(f"Unique food direction states: {len(all_combos)}")
print(f"Expected: 3^3 = 27 (but we only get a subset)")
print(f"\nAll combinations found:")
for combo in sorted(all_combos):
    print(f"  {combo}")

print(f"\n64 states = 8 danger combos (2^3) × 8 food combos (not 27!)")
print("The constraint: straight + left + right directions must sum properly")
print("\nThe issue: Using 3 orthogonal directions in 2D space creates dependencies")
