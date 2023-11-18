import torch
class Agent:
    def __init__(self):
        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.agent_position = (0, 0)

    def move(self, direction):
        new_position = (self.agent_position[0] + direction[0], self.agent_position[1] + direction[1])
        return new_position
    
class NeuralNetworkAgent(Agent):
    def __init__(self, neural_network):
        super().__init__()
        self.neural_network = neural_network

    def make_decision(self, observations):
        with torch.no_grad():
            observations_tensor = torch.tensor(observations, dtype=torch.float32)
            outputs = self.neural_network(observations_tensor)
            # Assuming outputs represent probabilities for each movement
            probabilities = torch.softmax(outputs, dim=0)
            movement_index = torch.multinomial(probabilities, 1).item()

            if movement_index == 0:
                return (-1, 0)  # Move up
            elif movement_index == 1:
                return (1, 0)   # Move down
            elif movement_index == 2:
                return (0, -1)  # Move left
            else:
                return (0, 1)   # Move right
