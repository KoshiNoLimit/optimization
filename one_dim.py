import numpy as np


def bi_section(f, L: (np.ndarray or float, np.ndarray or float), accuracy: float) -> (np.ndarray, int):
    """Метод бисекции"""
    a, b = L[0], L[1]
    vector = lambda: b - a
    x_mid = (a + b) / 2

    iterations = 0

    while True:
        iterations += 1

        x_left, x_right = a + vector()/4, b - vector()/4
        if f(x_left) < f(x_mid):
            b, x_mid = x_mid, x_left
        elif f(x_right) < f(x_mid):
            a, x_mid = x_mid, x_right
        else:
            a, b = x_left, x_right

        if np.linalg.norm(vector()) <= accuracy:
            return x_mid, iterations


def golden_section(f, L: (np.ndarray or float, np.ndarray or float), accuracy: float) -> (np.ndarray, int):
    """Метод золотого сечения"""
    gold_val = (3 - 5 ** 0.5) / 2  # 0.38196...
    a, b = L[0], L[1]
    vector = lambda: b - a

    x_left = a + gold_val * vector()
    x_right = b - gold_val * vector()

    iterations = 0

    while True:
        iterations += 1

        if f(x_left) < f(x_right):
            b = x_right
            x_right = x_left
            x_left = a + gold_val * vector()
        else:
            a = x_left
            x_left = x_right
            x_right = b - gold_val * vector()

        if np.linalg.norm(vector()) <= accuracy:
            return (a + b) / 2, iterations


def fibonacci_section(f, L: (np.ndarray or float, np.ndarray or float), accuracy: float) -> (np.ndarray, int):
    """Метод последовательности Фибоначи"""
    a, b = L[0], L[1]
    vector = lambda: b - a

    fib = [1, 1]
    length = np.linalg.norm(vector())
    while fib[-1] < length / accuracy:
        fib.append(fib[-1] + fib[-2])
    if len(fib) < 3:
        return (b + a) / 2, 0
    x_left = a + (fib[-3] / fib[-1]) * vector()
    x_right = a + (fib[-2] / fib[-1]) * vector()

    k = 1

    while True:
        x_left_last = x_left
        a_last, b_last = a, b

        if f(x_left) <= f(x_right):
            b = x_right
            x_right = x_left
            x_left = a + (fib[-(k + 2)] / fib[-(k)]) * vector()
        else:
            a = x_left
            x_left = x_right
            x_right = a + (fib[-(k + 1)] / fib[-(k)]) * vector()

        if k < len(fib) - 3:
            k += 1
        else:
            x_left = x_left_last
            x_right = x_left

            if f(x_left) < f(x_right):
                return (a_last + x_right) / 2, k - 1
            else:
                return (x_left + b_last) / 2, k - 1


def quadratic_interpolation(f, x1: float, step: float, accuracy: float) -> (float, int):
    """Квадратичная интерполяция"""
    iterations = 0

    while True:
        iterations += 1

        x2 = x1 + step

        if f(x1) > f(x2):
            x3 = x1 + 2 * step
        else:
            x3 = x1 - 2 * step

        points = tuple((x, f(x)) for x in (x1, x2, x3))
        point_min = min(points, key=lambda point: point[1])

        numerator = sum([(points[(i + 1) % 3][0] ** 2 - points[(i + 2) % 3][0] ** 2) * points[i][1] for i in range(3)])
        denominator = sum([(points[(i + 1) % 3][0] - points[(i + 2) % 3][0]) * points[i][1] for i in range(3)])
        if denominator == 0:
            x1 = point_min[0]
            continue
        new_x = 0.5 * (numerator / denominator)
        new_y = f(new_x)

        x_is_near = abs((point_min[0] - new_x) / new_x) < accuracy
        y_is_near = abs((point_min[1] - new_y) / new_y) < accuracy

        if x_is_near and y_is_near:
            return new_x, iterations
        elif points[0][0] <= new_x <= points[2][0]:
            x1 = min([point_min, (new_x, new_y)], key=lambda p: p[1])[0]
        else:
            x1 = new_x
        step /= 2


def cubic_interpolation(f, diff, x0: float, step: float, accuracy: float) -> (float, int):
    """Кубическая интерполяция"""
    step *= (-1) ** float(diff(x0) >= 0)
    x = x0

    while True:
        x += step

        if diff(x) * diff(x - step) <= 0:
            break

    x2, x1 = x, x - step

    iterations = 0

    while True:
        iterations += 1

        y1, y2 = f(x1), f(x2)
        d1, d2 = diff(x1), diff(x2)

        z = 3 * (y1 - y2) / (x2 - x1) + d1 + d2
        w = (-1) ** int(x1 >= x2) * (z ** 2 - d1 * d2) ** 0.5
        mu = (d2 + w - z) / (d2 - d1 + 2 * w)

        if mu < 0:
            x_min = x2
        elif mu <= 1:
            x_min = x2 - mu * (x2 - x1)
        else:
            x_min = x1

        while f(x_min) > f(x1):
            x_min -= 0.5 * (x_min - x1)

        if diff(x_min) < accuracy and abs((x_min - x1) / x_min) < accuracy:
            # print('iterations:', k)
            return x_min, iterations

        if diff(x_min) * diff(x1) < 0:
            x2, x1 = x1, x_min
        elif diff(x_min) * diff(x2) < 0:
            x1 = x_min
