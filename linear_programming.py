import numpy as np
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


class InfException(Exception):
    def __init__(self):
        super().__init__()
        self.txt = 'Infinitive function'


def gauss_forward(A0, f0):
    """Прямой ход метода Гаусса"""
    A = np.copy(A0)
    f = np.copy(f0)

    for i in range(0, len(A) - 1):
        for j in range(i + 1, len(A)):
            if A[i, i] == 0:
                raise KeyError('can not to solve with zero')
            coef = (A[j][i] / A[i][i])
            f[j] -= f[i] * coef
            A[j] -= A[i] * coef
    return A, f


def gauss_reverse(A0, f0):
    """Обратный ход метода Гаусса"""
    A = np.copy(A0)
    f = np.copy(f0)

    for i in range(0, len(A)):
        f[i] /= A[i][i]
        A[i] /= A[i][i]

    for i in range(len(A)-1, 0, -1):
        for j in range(i):
            f[j] -= f[i] * A[j][i]
            A[j] -= A[i] * A[j][i]

    return A, f


def gauss(A0, f0):
    """Метод Гаусса"""
    A1, f1 = gauss_forward(A0, f0)
    return gauss_reverse(A1, f1)


def initial_basic_solution1(A: np.matrix, b: np.ndarray, c: np.ndarray) -> (np.matrix, np.ndarray, np.ndarray, np.ndarray):
    """Выбор начального базиса методом Гаусса"""
    return *gauss(A, b), c, np.arange(A.shape[0])


def initial_basic_solution2(A: np.matrix, b: np.ndarray, c: np.ndarray) -> (np.matrix, np.ndarray, np.ndarray, np.ndarray):
    """Выбор начального базиса методом искусственного базиса"""
    big = 10**3

    A = np.concatenate((A, np.eye(A.shape[0])), axis=1)
    c = np.concatenate((c, np.full(A.shape[0], -big)))

    return A, b, c, np.arange(A.shape[1] - A.shape[0], A.shape[1])
    

def simplex(A: np.matrix, b: np.ndarray, c: np.ndarray, basis: np.ndarray) -> (np.matrix, np.ndarray, np.ndarray, np.ndarray, int):
    """Симплекс-метод"""
    iterations = 0

    no_basis = np.arange(A.shape[1])[np.logical_not(np.isin(np.arange(A.shape[1]), basis))] 

    while iterations < 10:
        iterations += 1

        d = np.array([np.sum(c[basis]*A[:, no_b]) - c[no_b] for no_b in no_basis])

        q = d.argmin()
        if d[q] >= 0:
            break

        alpha = np.linalg.solve(A[:, basis], A[:, no_basis][:, q])
        betta = np.linalg.solve(A[:, basis], b)

        ab = betta/alpha
        p, minimal = None, np.inf

        for i in np.arange(ab.size):
            if alpha[i] > 0 and ab[i] < minimal:
                p, minimal = i, ab[i]
        
        if p is None:
            raise InfException

        basis[p], no_basis[q] = no_basis[q], basis[p]
        no_basis = np.sort(no_basis)

        b[p] /= A[p, basis[p]]
        A[p] /= A[p, basis[p]]

        temp = int(p == 0)

        b[temp] -= b[p] * A[temp, basis[p]]
        A[temp] -= A[p] * A[temp, basis[p]]
    
    return A, b, c, basis, iterations


def two_phase(A: np.matrix, b: np.ndarray, c: np.ndarray) -> (np.matrix, np.ndarray, np.ndarray, np.ndarray, int):
    """Двухфазный симплекс-метод"""
    A, b, c, basis = initial_basic_solution2(A, b, c)
    c_temp = np.array([-1 for _ in range(A.shape[1]-A.shape[0])] + [0 for _ in range(A.shape[0])])
    A, b, c_temp, basis, iterations1 = simplex(A, b, c_temp, basis)
    A, b, c, basis, iterations2 = simplex(A, b, c, basis)
    return A, b, c, basis, iterations1 + iterations2


def branch_boundary(A: np.matrix, b: np.ndarray, c: np.ndarray) -> (np.matrix, np.ndarray, np.ndarray, np.ndarray):
    """Метод ветвей и границ"""

    def branch(A, b, c, id):
        try:
            _, b1, _, basis, _ = simplex(*initial_basic_solution1(A, b, c))
        except InfException:
            return None

        print('Branch: {}, result: {},   coordinates: {}'.format(id, np.sum(b1 * c[basis]), b1))

        b1 = np.round(b1, 7)
        if np.all(b1 == b1.astype(int)):
            return A, b1, c, basis

        else:
            float_index = 0
            while float_index < b1[float_index] == int(b1[float_index]):
                float_index += 1

            AA = np.concatenate((
                np.vstack((A, np.eye(c.size)[basis[float_index]])),
                np.eye(A.shape[0] + 1)[:, -1:]
            ), axis=1)
            first = branch(
                AA,
                np.append(b, int(b1[float_index])),
                np.append(c, 0),
                id + '1'
            )

            AA = np.concatenate((
                np.vstack((A, -np.eye(c.size)[basis[float_index]])),
                np.eye(A.shape[0] + 1)[:, -1:]
            ), axis=1)
            second = branch(
                AA,
                np.append(b, -int(b1[float_index] + 1)),
                np.append(c, 0),
                id + '2'
            )

            if not second:
                return first
            elif not first:
                return second
            else:
                if sum(first[2][first[3]] * first[1]) > sum(second[2][second[3]] * second[1]):
                    return first
                else:
                    return second

    return branch(A, b, c, '0')
