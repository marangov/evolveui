from NeuralNetwork import NeuralNetwork
import random
import copy
import torch

class GeneticAlgorithm:
    def __init__(self, population_size, fitness_function, mutation_rate=0.15, crossover_rate=0.65):
        self.population_size = population_size
        self.fitness_function = fitness_function
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.population = []
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.fitness_values = []
        self.fitness_history = []

    def generate_individual(self):
        neural_network = NeuralNetwork(11, 4)  # Adjust input and output sizes as needed
        return neural_network

    def initialize_population(self):
        self.population = [self.generate_individual() for _ in range(self.population_size)]

    def evaluate_population(self):
        self.fitness_values = [self.fitness_function(individual) for individual in self.population]
        best_index = max(range(self.population_size), key=lambda i: self.fitness_values[i])
        self.best_individual = self.population[best_index]
        self.best_fitness = self.fitness_values[best_index]

    def select_parents(self):
        parents = []
        total_fitness = sum(self.fitness_values)
        parents.append(self.best_individual)
        parents.append(self.best_individual)
        parents.append(self.best_individual)
        parents.append(self.best_individual)
        parents.append(self.best_individual)

        while len(parents) < self.population_size:
            r = random.uniform(0, total_fitness)
            cumulative_fitness = 0
            for i, fitness in enumerate(self.fitness_values):
                cumulative_fitness += fitness
                if cumulative_fitness >= r:
                    parents.append(self.population[i])
                    break
        return parents

    def crossover(self, parent1, parent2):
            if random.random() < self.crossover_rate:
                # Create copies of parents
                child1 = copy.deepcopy(parent1)
                child2 = copy.deepcopy(parent2)

                # Get parameters of the neural networks
                params1 = parent1.get_parameters()
                params2 = parent2.get_parameters()

                # Swap parameters at a random crossover point
                for key in params1.keys():
                    if random.random() < 0.5:
                        params1[key], params2[key] = params2[key], params1[key]

                # Set the parameters for the children
                child1.set_parameters(params1)
                child2.set_parameters(params2)


                return child1, child2
            return parent1, parent2

    def mutate(self, individual):
        mutated_individual = copy.deepcopy(individual)

        # Get the parameters of the neural network
        params = mutated_individual.get_parameters()

        for key in params.keys():
            if random.random() < self.mutation_rate:
                # Add some random noise to the parameter
                mutation = torch.randn_like(params[key])
                params[key] = params[key] + mutation

        mutated_individual.set_parameters(params)
        return mutated_individual

    def evolve(self):
        parents = self.select_parents()
        new_population = []
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            new_population.extend([child1, child2])
        self.population = new_population

    def run(self, generations):
        self.initialize_population()
        for generation in range(generations):
            GEN = generation
            self.evaluate_population()
            print(f"Generation {generation}: Best Fitness = {self.best_fitness}")
            self.evolve()
            self.fitness_history.append(self.best_fitness)
            with open('static\data.txt', 'w') as file:
                file.write(f"{generation} - {self.best_fitness}\n")