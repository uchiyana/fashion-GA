import random
import string
import argparse
from itertools import zip_longest


class DNA:
    def __init__(self, size):
        self.genes = random.choices(
            string.ascii_letters + string.digits + string.whitespace, k=size)

    def calc_fitness(self, target):
        self.fitness = sum([(g == t)
                            for g, t in zip_longest(self.genes, target)])

    def mutate(self, mutation_rate):
        for i in range(len(self.genes)):
            if(random.random() < mutation_rate):
                self.genes[i] = random.choice(
                    string.ascii_letters + string.digits + string.whitespace)

    def crossover(self, partner):
        midpoint = random.randrange(0, len(self.genes)-1)
        child = DNA(len(self.genes))
        child.genes = self.genes[:midpoint] + partner.genes[midpoint:]
        return child


class Population:
    def __init__(self, target, mutation_rate, population_size):
        self.target = target
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.population = [DNA(len(target)) for i in range(population_size)]
        self.mating_pool = list()

        self.calc_fitness()
        self.generations = 0

    def calc_fitness(self):
        for genes in self.population:
            genes.calc_fitness(self.target)

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
        normalized_best_fitness = best_fitness / len(self.target)
        print(
            f"{self.generations} fitness: {normalized_best_fitness:.2f} result: {''.join(best_dna.genes)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--target", default="To be or not to be", help="target string")
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

    population = Population(target, mutation_rate, population_size)

    for i in range(generation_size):
        population.natural_selection()
        population.generate()
        population.calc_fitness()
        population.display_best()


if __name__ == "__main__":
    main()
