import numpy as np
from multi_dim import nelder_mead


def penalty_functions(f, gs: np.ndarray, x: np.ndarray, accuracy: float, r=1, C=5) -> (np.ndarray, int):
    """Метод штрафных функций"""
    iterations = 0
    x = np.copy(x)

    penalty = lambda g_res: 0 if g_res <= 0 else g_res

    while True:
        iterations += 1

        P = lambda x: (r / 2) * sum([penalty(g(x)) for g in gs])
        F = lambda x: f(x) + P(x)

        x = nelder_mead(F, x, 2, 0.0001)[0]

        if P(x) < accuracy:
            break
        else:
            r *= C

    return x, iterations


def barrier_functions(f, gs: np.ndarray, x: np.ndarray, accuracy: float, r=1, C=10) -> (np.ndarray, int):
    """Метод барьерных функций"""
    iterations = 0
    x = np.copy(x)

    while True:
        iterations += 1

        P = lambda x: -r * sum([1/g(x) for g in gs])
        F = lambda x: f(x) + P(x)

        x = nelder_mead(F, x, 2, 0.0001)[0]

        if P(x) < accuracy:
            break
        else:
            r /= C

    return x, iterations


def penalty_barrier_functions(f, gs: np.ndarray, x: np.ndarray, accuracy: float, r=1, C=5) -> (np.ndarray, int):
    """Метод, взаимодополнения штрафных и барьерных функций"""
    if any([g(x) > 0 for g in gs]):
        return penalty_functions(f, gs, x, accuracy, r, C)
    else:
        return barrier_functions(f, gs, x, accuracy, r, C)


def lagrange_mody_functions(f, gs: np.ndarray, x: np.ndarray, accuracy: float, mu=None, r=3., C=5.) -> (np.ndarray, int):
    """Метод модифицированных функций Лагранжа"""
    if mu is None:
        mu = np.zeros((len(gs)))
    
    iterations = 0
    x = np.copy(x)

    penalty = lambda g_res: 0 if g_res <= 0 else g_res

    while True:
        if iterations > 100:
            break

        iterations += 1

        P = lambda x: sum([max([0] + mu[i] + r*penalty(gs[i](x)))**2 - mu[i]**2 for i in range(len(gs))]) / (2*r)
        L = lambda x: f(x) + P(x)
            
        x = nelder_mead(L, x, 2, 0.0001)[0]

        if P(x) < accuracy:
            break
        else:
            
            mu += np.array(list(map(lambda g: r*penalty(g(x)), gs)))
            mu[mu < 0] = 0

            r *= C
    
    return x, iterations
