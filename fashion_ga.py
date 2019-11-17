import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from itertools import zip_longest
import random
import string
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class FashionNN:
    def __init__(self, shape):
        self.shape = shape
        model_file = "fashion.h5"
        if os.path.exists(model_file):
            self.model = keras.models.load_model('fashion.h5')
        else:
            self.model = self.generate_model()

    def create_model(self):
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=self.shape),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def generate_trained_model(self):
        model = self.create_model()
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images,
                                       test_labels) = fashion_mnist.load_data()
        model.fit(train_images, train_labels,  epochs=5)
        model.save('fashion.h5')
        return model

    def predict(self, img):
        return self.model.predict(img)


class DNA:
    def __init__(self, shape):
        self.genes = np.random.randint(0, 255, size=shape)

    def calc_fitness(self, fashion_nn, type_index):
        self.fitness = fashion_nn.predict(
            (np.expand_dims(self.genes, 0)))[0][type_index]

    def mutate(self, mutation_rate):
        mask = np.random.rand(*self.genes.shape) < mutation_rate
        r = np.random.randint(0, 255, self.genes.shape)
        self.genes[mask] = r[mask]

    def crossover(self, partner):
        index_a = np.arange(0, self.genes.size).reshape(self.genes.shape)
        midpoint = random.randrange(0, self.genes.size)
        mask = index_a > midpoint
        child = DNA(self.genes.shape)
        child.genes = self.genes.copy()
        child.genes[mask] = partner.genes[mask]
        return child


class Population:
    def __init__(self, shape, type_index, mutation_rate, population_size):
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.population = [DNA(shape) for i in range(population_size)]
        self.mating_pool = list()
        self.fashion_nn = FashionNN(shape)
        self.type_index = type_index

        self.calc_fitness()
        self.generations = 0

    def calc_fitness(self):
        for genes in self.population:
            genes.calc_fitness(self.fashion_nn, self.type_index)

    def natural_selection(self):
        self.mating_pool.clear()
        max_fitness = float(max([g.fitness for g in self.population]))
        for g in self.population:
            num = int(100 * (g.fitness / max_fitness))
            self.mating_pool += [g for i in range(num)]

    def __make_child(self):
        a = random.randrange(len(self.mating_pool))
        b = random.randrange(len(self.mating_pool))
        parent_a = self.mating_pool[a]
        parent_b = self.mating_pool[b]
        child = parent_a.crossover(parent_b)
        child.mutate(self.mutation_rate)
        return child

    def generate(self):
        self.population = [self.__make_child()
                           for i in range(self.population_size)]
        self.generations += 1

    def get_best(self):
        result = [g.fitness for g in self.population]
        best_fitness = max(result)
        best_index = result.index(best_fitness)
        best_dna = self.population[best_index]
        return (best_fitness, best_dna)

    def display_best(self):
        best_fitness, best_dna = self.get_best()
        print(
            f"{self.generations} fitness: {best_fitness:.2f} {best_dna.genes.shape}")

    def display_best_image(self):
        best_fitness, best_dna = self.get_best()
        plt.close()
        plt.figure()
        plt.imshow(best_dna.genes)
        plt.colorbar()
        plt.grid(False)
        plt.pause(0.001)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--target", type=int, default=0,
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        help="0: t-shirt, 1: trouser, 2: pullover, 3: dress, 4: coat, 5: sandal, 6: shirt, 7: sneaker, 8: bag, 9: boot")
    parser.add_argument(
        "-m", "--mutation", type=float, default=0.01, help="mutation rate")
    parser.add_argument(
        "-p", "--population", type=int, default=100, help="population size")
    parser.add_argument(
        "-g", "--generation", type=int, default=100, help="generation size")

    args = parser.parse_args()
    target = args.target
    mutation_rate = args.mutation
    population_size = args.population
    generation_size = args.generation
    size = (28, 28)

    population = Population(size, target, mutation_rate, population_size)
    for i in range(generation_size):
        population.natural_selection()
        population.generate()
        population.calc_fitness()
        population.display_best()
        population.display_best_image()


if __name__ == "__main__":
    main()
