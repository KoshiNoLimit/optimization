import numpy as np
import random
from copy import deepcopy
from multi_dim import hook_jeeves
from one_dim import bi_section


class Cluster:
    def __init__(self, points, f):
        self.points = points
        self.f = f

    def __str__(self):
        return str(self.points)

    def __repr__(self):
        return str(self.points)

    def update(self, other):
        self.points += other.points

    def distance(self, other):
        return min([np.linalg.norm(p_s - p_o) for p_s in self.points for p_o in other.points])

    @property
    def middle(self):
        return min(self.points, key=lambda x: self.f(x))


def nearest_neighbour(f, points, accuracy):
    """Кластеризация методом наиближайшего соседа"""
    clusters = [Cluster([point, ], f) for point in points]

    while len(clusters) > 1:
        pair = min([(i, j, clusters[i].distance(clusters[j]))
                    for i in range(len(clusters)-1)
                    for j in range(i+1, len(clusters))],
                   key=lambda x: x[2])

        if pair[2] > accuracy:
            break

        clusters[pair[0]].update(deepcopy(clusters[pair[1]]))
        del clusters[pair[1]]

    return clusters


def competing_points(f, alpha, betta, accuracy):
    """Метод конкурирующих точек"""
    points = np.array([
        np.array([random.uniform(alpha, betta) for _ in range(2)])
        for _ in range(80)
    ])


    iterations = 0
    while len(points) > 1:
        iterations += 1

        print('Iteration {},   points: {},   best: {}'.format(iterations, len(points), min(points, key=lambda x: f(x))))
        points = [hook_jeeves(f, point, 0.01, 0.5, bi_section, maxiter=1)[0] for point in points]
        clusters = nearest_neighbour(f, points, accuracy)

        if len(clusters) == len(points):
            accuracy *= 2
        else:
            points = [cluster.middle for cluster in clusters]

    print('Iterations:', iterations, end='   ')

    return hook_jeeves(f, points[0], 0.0001, 0.3, bi_section)[0]


def genetic(f, dim, alpha, betta, gs=None, tries=3, population_size=50, population=None, maxiter=100, F=0.05, P=0.7):
    """Генетический алгоритм"""
    result = np.zeros(dim)

    population_count = 0
    while population_count < tries:
        population_count += 1

        if population is None:
            population = np.array([
                np.array([random.uniform(alpha, betta) for _ in range(dim)])
                for _ in range(population_size)
            ])

        iterations = 0
        while iterations < maxiter:
            iterations += 1

            population = sorted(list(population), key=lambda x: f(x))
            population = np.array(population[:population_size])

            np.random.shuffle(population)
            childs = []
            for i in np.arange(0, len(population), 2):
                x = population[i] * P + population[i + 1] * (1 - P)
                y = population[i] * (P - 1) + population[i + 1] * P
                if gs:
                    if all([g(x) <= 0 for g in gs]):
                        childs.append(x)
                    if all([g(y) <= 0 for g in gs]):
                        childs.append(y)
                else:
                    childs.append(x)
                    childs.append(y)

            population = np.concatenate((population, np.array(childs)))

            mutants = []
            for i in np.arange(0, int(population_size * F)):
                r = random.uniform(0, 1)
                if r > F:
                    mutant = population[i]
                    mutant[random.randint(0, dim - 1)] = random.uniform(alpha, betta)
                    mutants.append(mutant)

            if len(mutants):
                population[max(range(len(population)), key=lambda x: f(population[x]))] = min(mutants,  key=lambda x: f(x))

        the_best = min(population, key=lambda x: f(x))
        if f(the_best) < f(result):
            result = the_best

    return result
