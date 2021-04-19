import numpy as np
from multi_dim_conditional import penalty_barrier_functions
from multi_dim import hook_jeeves
from random import random
from multi_extrema import genetic
from one_dim import bi_section


def convolution_criteria(fs, gs, dim=3):
    """Метод свертки критериев"""
    ideal = np.array([hook_jeeves(f, np.zeros(dim), 0.01, 0.5, bi_section)[0] for f in fs])

    front = []

    for _ in np.arange(10):
        r = random()
        W = np.array([r, 1 - r])
        F = lambda x: sum(W * np.array([(fs[i](ideal[i]) - fs[i](x)) for i in np.arange(len(ideal))])**2)
        X = penalty_barrier_functions(F, gs, np.zeros(dim), 0.0001)[0]
        front.append((W, X))

    return front, ideal


def genetic_convolution_criteria(fs, gs, dim=3, population=None):
    """Генетическая вариация алгоритма"""
    ideal = np.array([hook_jeeves(f, np.zeros(dim), 0.001, 0.5, bi_section)[0] for f in fs])

    front = []

    for _ in np.arange(10):
        r = random()
        W = np.array([r, 1 - r])
        F = lambda x: sum(W * np.array([(fs[i](ideal[i]) - fs[i](x)) for i in np.arange(len(ideal))])**2)
        X = genetic(F, 3, -2, 4, gs=gs, tries=2, population=population)
        front.append((W, X))

    return front, ideal
