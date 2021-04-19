import numpy as np
from one_dim import bi_section, golden_section, fibonacci_section


def hook_jeeves(f, x: np.ndarray, accuracy: float, step: float, method, maxiter=300) -> (np.ndarray, int):
    """Метод Хука-Дживса"""
    x = np.copy(x)
    best_x = np.copy(x)

    E = np.eye(len(x))

    iterations = 0

    while step > accuracy/100 and iterations < maxiter:
        iterations += 1

        for i in np.arange(x.size):
            if f(x + E[i] * step) < f(x):
                x += E[i] * step
            elif f(x - E[i] * step) < f(x):
                x -= E[i] * step

        if (x != best_x).any():
            best_x, _ = method(f, [best_x, x], accuracy)
            x = np.copy(best_x)
        else:
            step /= 2

    return best_x, iterations


def nelder_mead(f, x: np.ndarray, step: float, accuracy: float, alpha=2, betta=0.5, gamma=2, iters=300) -> (np.ndarray, int):
    """Метод Нэлдера-Мида"""
    dim = len(x)
    xs = [np.copy(x) for _ in range(dim + 1)]
    for i in range(dim):
        for j in range(dim):
            xs[i][j] += (step / (dim * 2 ** 0.5)) * ((dim + 1) ** 0.5 - 1 + int(i == j) * dim)

    xs.sort(key=lambda x: f(x))
    k = 0
    while k < iters:
        k += 1
        x_mid = np.sum(xs, axis=0) / (dim + 1)

        # Отражение
        x_r = x_mid + alpha * (x_mid - xs[-1])

        # Оценка точки
        if f(x_r) < f(xs[0]):
            x_e = x_mid + gamma * (x_r - x_mid)
            if f(x_e) < f(x_r):
                xs[-1] = x_e
            else:
                xs[-1] = x_r
        elif f(x_r) < f(xs[-2]):
            xs[-1] = x_r
        else:
            if f(x_r) < f(xs[-1]):
                xs[-1] = x_r

            # Сжатие
            x_s = x_mid + betta * (xs[-1] - x_mid)
            if f(x_s) < f(xs[-1]):
                xs[-1] = x_s
            else:
                for i in range(1, dim + 1):
                    xs[i] = (xs[i] + xs[0]) / 2

        xs.sort(key=lambda x: f(x))

        if np.linalg.norm(xs[0] - xs[1]) < accuracy \
                and np.linalg.norm([(f(xs[i]) - f(xs[0])) for i in np.arange(1, dim + 1)]) < (dim + 1) * accuracy:
            break
    return xs[0], k


def gradient_descent(f, grad, x: np.ndarray, accuracy: float, step=1, **kwargs) -> (np.ndarray, int):
    """Метод градиентного спуска"""
    iterations = 0

    x_new, x_last = np.copy(x), np.copy(x)
    d = - grad(x_last)

    while iterations < 10000:
        iterations += 1

        x_new, _ = bi_section(f, [x_last, x_last + d*(step/np.linalg.norm(d))], accuracy/10)

        if np.linalg.norm(grad(x_new)) < 100*accuracy:
            break
        else:
            x_last = np.copy(x_new)
            d = - grad(x_last)

    return x_new, iterations
    

def conjugate_gradients(f, grad, x: np.ndarray, accuracy: float, formula='Fletcher-Reeves', step=1, **kwargs) -> (np.ndarray, int):
    """Метод сопряженного градиента"""
    iterations = 0

    x_new, x_last = np.copy(x), np.copy(x)
    d = - grad(x_new)

    while iterations < 10000:
        iterations += 1

        x_new, _ = bi_section(f, [x_last, x_last + d*(step/np.linalg.norm(d))], accuracy/10)
        
        if np.linalg.norm(grad(x_new)) < 100*accuracy:
            break
        else:
            g_last, g_new = grad(x_last), grad(x_new)
            if formula == 'Fletcher-Reeves':
                w = (np.linalg.norm(g_new)**2 / np.linalg.norm(g_last)**2)
            elif formula == 'Polak-Ribier':
                w = (np.matmul(g_new, g_new-g_last) / np.linalg.norm(g_last)**2)
            else:
                raise ValueError('have not this formula')

            x_last = np.copy(x_new)
            d = - grad(x_new) + w*d

    return x_new, iterations


def davidon_fletcher_powell(f, grad, x: np.ndarray, accuracy: float, step=1, **kwargs) -> (np.ndarray, int):
    """Метод Дэвидона-Флэтчера-Пауэлла"""
    iterations = 0

    x_new, x_last = np.copy(x), np.copy(x)
    G = np.eye(x.size)
    d = - np.matmul(G, grad(x_new))
    x_new, _ = bi_section(f, [x_last, x_last + d * (step / np.linalg.norm(d))], accuracy/10)

    while iterations < 10000:
        iterations += 1

        g = grad(x_new) - grad(x_last)
        x_delta = x_new - x_last
        G += np.linalg.norm(x_delta)/np.matmul(x_delta, g) - np.linalg.norm(np.matmul(G, g))/np.matmul(g, np.matmul(G, g))

        d = - np.matmul(G, grad(x_new))
        x_last = np.copy(x_new)
        x_new, _ = bi_section(f, [x_last, x_last + d * (step / np.linalg.norm(d))], accuracy/10)

        if np.linalg.norm(grad(x_new)) < 100*accuracy:
            break

        if iterations % 100 == 0:
            G = np.eye(x.size)
            d = - np.matmul(G, grad(x_new))
            x_new, _ = bi_section(f, [x_last, x_last + d * (step / np.linalg.norm(d))], accuracy/10)

    return x_new, iterations


def levenberg_marquardt(f, grad, x: np.ndarray, accuracy: float, hesse, mu=100., **kwargs) -> (np.ndarray, int):
    """Метод Левенберга-Марквардта"""
    x_new, x_last = np.copy(x), np.copy(x)
    iterations = 0
    E = np.eye(x.size)

    while np.linalg.norm(grad(x_new)) > 100*accuracy and iterations < 10000:
        iterations += 1

        H = hesse(x_new)

        while True:
            x_new = x_last - np.matmul(np.linalg.inv(H + mu*E),  grad(x_last))
            if f(x_new) < f(x_last):
                mu /= 2.
                x_last = np.copy(x_new)
                break 
            else:
                mu *= 2.
       
    return x_new, iterations
