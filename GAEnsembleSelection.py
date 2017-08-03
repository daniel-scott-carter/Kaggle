from random import randint
from operator import itemgetter
from random import uniform
import Ensemble as en


# HYPER PARAMS
crossover_prob = 0.8
mutation_prob = 1.0
population_num = 0
generation_num = 0
chromosomes_size = 0
model_list = []

predictors = []
classifications = []

chromosomes = []
best_overall_chromosome = []


def setup(list_of_models, population_size, number_of_generations, test_set, test_set_classifications):
    global model_list
    global population_num
    global generation_num
    global predictors
    global classifications
    global chromosomes_size

    model_list = list_of_models
    population_num = population_size
    generation_num = number_of_generations
    predictors = test_set
    classifications = test_set_classifications
    chromosomes_size = len(list_of_models)
    print("GA Setup")

def initialise_first_population():
    global chromosomes_size
    global chromosomes
    global population_num

    for i in range(population_num):
        valid_chromosome = False

        while not valid_chromosome:
            current_chromosome = []
            for j in range(chromosomes_size):
                current_chromosome.extend([randint(0, 1)])

            if 1 in current_chromosome:
                valid_chromosome = True

        chromosomes.append(current_chromosome)


def get_fitness_of_each_chromosome():
    global model_list
    global chromosomes
    global predictors
    global classifications
    #global chromosome_fitness
    global population_num

    for chromosome in chromosomes:
        model_sublist = []
        for i in range(len(chromosome)):
            if chromosome[i] == 1:
                model_sublist.append(model_list[i])

        ''' This is gonna take some time. Look at optimising the ensemble code! '''
        chromosome_accuracy = en.get_ensemble_predict_fitness(model_sublist, predictors, classifications)
        chromosome.extend([chromosome_accuracy])


def sort_chromosomes_by_fitness():
    global chromosomes
    global chromosomes_size

    chromosomes = sorted(chromosomes, key=itemgetter(chromosomes_size))


def roulette_parent_selection():
    global chromosomes
    ''' Calculate the size of our roulette wheel based on total fitness '''
    roulette_size = sum(chromosome[len(chromosome)-1] for chromosome in chromosomes)

    ''' Roll two die on the roulette wheel to pick our two parents based on their fitness'''
    pick1 = uniform(0, roulette_size)
    pick2 = uniform(0, roulette_size)

    selected_chromosomes = []

    current_fitness_total = 0

    ''' Loop through each Chromosome and add up fitness totals, if we 
        reach a greater value than our pick then we select the previous chromosome '''
    for chromosome in chromosomes:
        current_fitness_total += chromosome[len(chromosome)-1]

        if current_fitness_total > pick1:
            selected_chromosomes.append(chromosome[0:len(chromosome) - 1])
            if len(selected_chromosomes) == 2:
                return selected_chromosomes
        if current_fitness_total > pick2:
            selected_chromosomes.append(chromosome[0:len(chromosome) - 1])
            if len(selected_chromosomes) == 2:
                return selected_chromosomes


def generate_new_population():
    global chromosomes
    global model_list
    global best_overall_chromosome
    global population_num
    global chromosomes_size

    new_population = []

    best_from_previous_gen = chromosomes[len(chromosomes)-1]
    second_best_previous_gen = chromosomes[len(chromosomes)-2]

    new_population.append(best_from_previous_gen[0:len(best_from_previous_gen) - 1])
    new_population.append(second_best_previous_gen[0:len(second_best_previous_gen) - 1])

    if len(best_overall_chromosome) == 0:
        best_overall_chromosome = best_from_previous_gen
    elif best_overall_chromosome[chromosomes_size] < best_from_previous_gen[chromosomes_size]:
        best_overall_chromosome = best_from_previous_gen

    population_iterations = range(int(population_num/2))

    for individual in population_iterations:

        parents = roulette_parent_selection()

        parent1 = parents[0][0:len(parents[0])]
        parent2 = parents[1][0:len(parents[0])]

        child1 = parent1
        child2 = parent2

        crossover_chance = uniform(0, 1)
        if crossover_chance < crossover_prob:

            child1 = []
            child2 = []

            chromosome_length = range(len(parent1))

            for gene in chromosome_length:

                chosen_parent_1 = randint(1, 2)
                chosen_parent_2 = randint(1, 2)

                if chosen_parent_1 == 1:
                    child1.extend([parent1[gene]])
                else:
                    child1.extend([parent2[gene]])

                if chosen_parent_2 == 1:
                    child2.extend([parent1[gene]])
                else:
                    child2.extend([parent2[gene]])

        mutation_chance = uniform(0, 1)

        if mutation_chance <= mutation_prob:

            rand_mutate1 = randint(0, chromosomes_size-1)
            rand_mutate2 = randint(0, chromosomes_size-1)

            if child1[rand_mutate1] == 1:
                child1[rand_mutate1] = 0
            else:
                child1[rand_mutate1] = 1

            if child2[rand_mutate2] == 1:
                child2[rand_mutate2] = 0
            else:
                child2[rand_mutate2] = 1

        if len(new_population) != population_num:
            new_population.append(child1)
            if len(new_population) != population_num:
                new_population.append(child2)

    chromosomes = new_population


def execute():
    global generation_num
    global best_overall_chromosome
    initialise_first_population()

    generations = range(generation_num)

    for generation in generations:

        get_fitness_of_each_chromosome()
        sort_chromosomes_by_fitness()
        generate_new_population()

    model_sublist = []
    for i in range(len(best_overall_chromosome[0:len(best_overall_chromosome) - 1])):
        if best_overall_chromosome[i] == 1:
            model_sublist.append(model_list[i])

    print("Best Ensemble: ")
    result = en.get_ensemble_predict_fitness(model_sublist, predictors, classifications)
    print(str(best_overall_chromosome))

    return model_sublist


