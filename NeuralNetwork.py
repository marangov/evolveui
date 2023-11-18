import torch
import torch.nn as nn
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)  # Adjust input size and hidden layer size as needed
        self.fch = nn.Linear(256 , 256)
        self.fch2 = nn.Linear(256 , 256)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fch(x)
        x = self.fch2(x)
        x = self.fc2(x)
        return x
    
    def get_parameters(self):
        return self.state_dict()
    
    def set_parameters(self, params):
        self.load_state_dict(params)