from flask import Flask, render_template, request
import json
from Enviroment import Environment
from Game import Game
from Agent import NeuralNetworkAgent 
from GeneticAlgorithm import GeneticAlgorithm
import numpy as np
import matplotlib.pyplot as plt
import imageio
import numpy as np
import imageio
import time

app = Flask(__name__, template_folder='templates')
COUNTER = 0
GEN = 0
@app.route('/')
def index():
    return render_template('grid_clicker.html')

def create_bigger_grid(state, target_size=1000):
  """Creates a bigger grid from a smaller grid with the same elements.

  Args:
    state: A NumPy array representing the smaller grid.
    target_size: The desired size of the bigger grid.

  Returns:
    A NumPy array representing the bigger grid.
  """

  # Calculate the scaling factor.
  scaling_factor = target_size / state.shape[0]

  # Create a bigger grid.
  bigger_grid = np.zeros((target_size, target_size, state.shape[2]), dtype=state.dtype)

  # Copy the elements of the smaller grid to the bigger grid and add a 10px circle
  # around each element.
  for i in range(state.shape[0]):
    for j in range(state.shape[1]):
      bigger_grid[int(i * scaling_factor):int(i * scaling_factor + 11),
                   int(j * scaling_factor):int(j * scaling_factor + 11), :] = state[i, j, :]

  # Return the bigger grid.
  return bigger_grid
  

def create_gif_from_states(states, output_filename, duration=0.1):
    images = []

    for state in states:
        # Assuming state is a 3D NumPy array
        bigger_grid = create_bigger_grid(state)
        image = np.uint8(bigger_grid)

        # Resize the image to 1000x1000 pixels
        images.append(image)

    # Save the images as a GIF
    imageio.mimsave(output_filename, images, duration=duration)

def create_gif_from_states_normal(states, output_filename, duration=0.1):
    images = []

    for state in states:
        # Assuming state is a 3D NumPy array
        image = np.uint8(state)
        # Resize the image to 1000x1000 pixels
        images.append(image)
    # Save the images as a GIF
    imageio.mimsave(output_filename, images, duration=duration)


@app.route('/runAlgorithm', methods=['POST'])
def plus_plus():

    def play_game(agent , save_movements):
        grid_size = 30
        num_holes = 10
        env = Environment(grid_size, num_holes, specific_holes)
        game = Game(env, agent, save_movements)
        game.play(random_moves=True)

        return game.counter , game.final , game

    def fitness_function(individuo):
            a_perf = 0
            for _ in range(n_juegos):
                agente = NeuralNetworkAgent(neural_network=individuo)
                perf , win , crash = play_game(agente , False)
                if win:
                    a_perf += 20000
                    a_perf -= perf
                else:
                    if crash:
                        a_perf += perf
                        a_perf -= 10
                    else:
                        a_perf += perf
            #print(perf , '  ' , stop)
            return a_perf
    data = request.get_json()
    generations = int(data['generations'])
    population_size = int(data['populationSize'])
    n_juegos = int(data['nJuegos'])
    specific_holes = [(i[0] , i[1]) for i in json.loads(data['holes'])]
    
    ga = GeneticAlgorithm(population_size=population_size, fitness_function=fitness_function)
    ga.run(generations=generations)

    bnn = ga.best_individual
    agente = NeuralNetworkAgent(neural_network=bnn)

    grid_size = 30
    num_holes = 28
    env = Environment(grid_size, num_holes, specific_holes)
    print(env.specific_holes)
    game = Game(env, agente, True)
    game.play(random_moves=True)


    print(game.counter)
    print(game.final)
    print(game.environment.holes)
    print(len(game.game_movie))
    print(game.environment.specific_holes)
    print(game.crash)

    create_gif_from_states(game.game_movie, 'static\output.gif', duration=2)
    create_gif_from_states_normal(game.game_movie , 'static\output2.gif' , duration = 2)
    
    return str(ga.best_fitness)

# @app.route('/update_counter', methods=['POST'])
# def update_counter():
#     global COUNTER
#     COUNTER += 3
#     global GEN
#     print(GEN)

#     time.sleep(2)
#     return str(GEN)

if __name__ == '__main__':
    app.run()   