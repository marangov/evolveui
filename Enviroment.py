import random

class Environment:

    def __init__(self, grid_size, num_holes , specific_holes):
        self.grid_size = grid_size
        self.start_position = (0, 0)
        self.goal_position = (grid_size - 1, grid_size - 1)
        self.agent_position = (1, 1)
        self.specific_holes = specific_holes
        self.holes = self.generate_random_holes(num_holes)
        self.holes = self.specific_holes + self.holes  # Combine specific holes and random holes
        random.shuffle(self.holes)  # Shuffle the combined list

    def generate_random_holes(self, num_holes):
        if num_holes >= 2 and num_holes <= self.grid_size - 2:
            possible_coordinates = [(x, y) for x in range(1, self.grid_size - 1) for y in range(1, self.grid_size - 1)]
            random_holes = random.sample(possible_coordinates, num_holes)
            return random_holes
        else:
            raise ValueError("Number of holes should be between 2 and grid_size - 2")
        
