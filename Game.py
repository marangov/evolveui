import numpy as np
import copy
class Game:
    def __init__(self, environment, agent , save_movements):
        self.environment = environment
        self.agent = agent
        self.counter = 0
        self.final = False
        self.save_movements = save_movements
        self.game_movie = []
        self.crash = False

    

    def get_observations(self):
        # Implement this method to get observations of the surrounding squares.
        agent_x, agent_y = self.agent.agent_position
        goal_x, goal_y = self.environment.goal_position
        observations = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                x = agent_x + dx
                y = agent_y + dy
                if 0 <= x < self.environment.grid_size and 0 <= y < self.environment.grid_size:
                    if (x, y) == self.environment.goal_position:
                        observations.append(1)  # 1 represents the goal
                    elif (x, y) in self.environment.holes:
                        observations.append(-1)  # -1 represents a hole
                    else:
                        observations.append(0)  # 0 represents an empty space
                else:
                    observations.append(-1)  # -1 represents a wall

        # Calculate the direction between the agent and the goal
        direction = np.array([goal_x - agent_x, goal_y - agent_y])
        norm = np.linalg.norm(direction)
        if norm == 0:
            observations.append(0)  # 0 if the agent is at the goal
        else:
            normalized_direction = direction / norm
            observations.append(normalized_direction[0])
            observations.append(normalized_direction[1])

        return observations

    def is_valid_move(self, new_position):
        return 0 <= new_position[0] < self.environment.grid_size and 0 <= new_position[1] < self.environment.grid_size

    def get_env_grid(self):
        grid = np.zeros((self.environment.grid_size, self.environment.grid_size, 3), dtype=np.uint8)
        start_color = (255, 0, 0)  # Red for start
        goal_color = (0, 255, 0)   # Green for goal
        agent_color = (0, 0, 255)  # Blue for agent
        hole_color = (255, 255, 255)  # White for holes

        grid[self.environment.start_position[0], self.environment.start_position[1]] = start_color
        grid[self.environment.goal_position[0], self.environment.goal_position[1]] = goal_color
        grid[self.agent.agent_position[0], self.agent.agent_position[1]] = agent_color

        for hole in self.environment.holes:
            grid[hole[0], hole[1]] = hole_color

        return grid

        # plt.imshow(grid)
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('Navigation problem on a {}x{} grid'.format(self.environment.grid_size, self.environment.grid_size))
        # plt.show()

    def play(self, random_moves=False):
        #while self.agent.agent_position != self.environment.goal_position:
        while self.counter <= 200 and self.agent.agent_position != self.environment.goal_position:
            t_pos = copy.deepcopy(self.agent.agent_position)
            if random_moves:
                observations = self.get_observations()  # Implement this method to get surrounding observations
                movement = self.agent.make_decision(observations)
            else:
                move = input("Enter a move (W for up, S for down, A for left, D for right): ").upper()

                if move not in ['W', 'S', 'A', 'D']:
                    print("Invalid move. Use W, S, A, or D.")
                    continue

                if move == 'W':
                    movement = (-1, 0)  # Move up
                elif move == 'S':
                    movement = (1, 0)   # Move down
                elif move == 'A':
                    movement = (0, -1)  # Move left
                elif move == 'D':
                    movement = (0, 1)   # Move right

            new_position = self.agent.move(movement)

            if self.is_valid_move(new_position):
                if new_position in self.environment.holes:
                    # print("Agent landed on a hole. Game over!")
                    self.crash = True
                    break
                self.agent.agent_position = new_position
            else:
                # print("Agent hit a wall. Game over!")
                self.crash = True
                break
            self.counter += 1
            self.environment.holes.append(t_pos)
            if self.save_movements == True:
                self.game_movie.append(self.get_env_grid())
            # plt.pause(0.1)
                        
        if self.agent.agent_position == self.environment.goal_position:
            self.final = True