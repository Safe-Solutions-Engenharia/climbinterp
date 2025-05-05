from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import warnings
import logging


'''
Classe especifica para interpolação para valores não negativos para x > 0, priorizando curvas conservadoras. Pode ser
utilizada como um módulo, sendo chamado em outros .py. (INCOMPLETA)
'''

warnings.filterwarnings("ignore")


def exp(x, a, b):
    return a * np.exp(b*x)

def logarithmic(x, a, b):
    return a * np.log(x) + b

def linear(x, a, b):
    return a * x + b

def linear_with_c_zero(x, a):
    return a * x

def exception_linear(x, a):
    return x * a


# Metade inicial > exponencial
# metade final > log ou linear

#TODO separar parte que faz as operações para diminuir dados da interpolação em si.

class ClimbInterp:
    def __init__(self, x_value, y_value, show_graph=False, graph_title=None):
        self.x_value = x_value
        self.y_value = y_value
        self.show_graph = show_graph
        self.graph_title = graph_title

        self.x_exp = None
        self.y_exp = None

        self.popt_exp = None
        self.popt_linear = None
        self.is_linear = False
        self.is_logarithmic = False

        self.intersection = None

        self.arrange_points()
        self.curve_fit_exp()
        self.curve_fit_linear()

    @staticmethod
    def _exp(x, a, b):
        return a * np.exp(b*x)

    @staticmethod
    def _logarithmic(x, a, b):
        return a * np.log(x) + b

    @staticmethod
    def _linear(x, a, b):
        return a * x + b

    @staticmethod
    def _exception_linear(x, a):
        return x * a

    @staticmethod
    def _r_squared(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)

    @staticmethod
    def _sort_points_by_x(x_p, y_p):
        sorted_points = sorted(zip(x_p, y_p), key=lambda point: point[0])
        sorted_x, sorted_y = zip(*sorted_points)
        return sorted_x, sorted_y

    def arrange_points(self):
        x_points, y_points = self._sort_points_by_x(self.x_value, self.y_value)

        x = [x_points[0]]
        y = [y_points[0]]
        for i in range(1, len(y_points)):
            if y[-1] > y_points[i]:
                pass
            else:
                x.append(x_points[i])
                y.append(y_points[i])

        result_dict = {}
        for idx, value in enumerate(x):
            if value not in result_dict:
                result_dict[value] = y[idx]
            else:
                result_dict[value] = max(result_dict[value], y[idx])

        self.x_value = list(result_dict.keys())
        self.y_value = list(result_dict.values())

    def _get_lower_error(self, x, y):
        if len(x) == 1:
            popt, pcov = optimize.curve_fit(self._exception_linear, [0, x[0]], [0, y[0]])
            return x, y, popt, True

        for index in range(0, len(x) - 1):
            try:
                x_fake, y_fake = x[0:index + 2], y[0:index + 2]
                popt, pcov = optimize.curve_fit(self._exp, x_fake, y_fake, maxfev=10000)
                fitted_y = self._exp(x_fake, *popt)

                # Cálculo do R²
                r_squared_value = self._r_squared(y_fake, fitted_y)

                if len(x) == 2:
                    return x, y, popt, False

                try:
                    x_new, y_new = x[0:index + 3], y[0:index + 3]
                    popt_new, pcov_new = optimize.curve_fit(self._exp, x_new, y_new, maxfev=10000)
                    fitted_y_new = self._exp(x_new, *popt_new)

                    # Cálculo do R²
                    r_squared_value2 = self._r_squared(y_new, fitted_y_new)

                    if abs(r_squared_value - r_squared_value2) < 0.01 and len(x) >= index + 3:
                        continue
                    else:
                        return x_fake, y_fake, popt, False

                except Exception as e:
                    #logging.warning(f'{e} - INNER EXCEPTION. (better)')
                    return x_fake, y_fake, popt, False
            except Exception as e:
                #logging.warning(f'{e} - OUTTER EXCEPTION. (worse)')
                if 'x_new' in locals() and 'y_new' in locals():
                    return x_new, y_new, popt_new, False
                elif index == 0:
                    x_fake_with_origin = [0] + x_fake
                    y_fake_with_origin = [0] + y_fake
                    try:
                        popt, pcov = optimize.curve_fit(self._exception_linear, x_fake_with_origin, y_fake_with_origin,
                                                        maxfev=10000)
                    except:
                        raise ValueError('Erro crítico na primeira parte da interpolação!')
                    return x_fake, y_fake, popt, True

    def curve_fit_exp(self):
        x = np.array(self.x_value)
        y = np.array(self.y_value)

        x, y, popt, is_linear = self._get_lower_error(x, y)

        new_x = []
        new_y = []
        for x_teste, y_teste in zip(self.x_value, self.y_value):
            if x_teste in x and y_teste in y:
                continue
            else:
                new_x.append(x_teste)
                new_y.append(y_teste)

        self.x_exp = np.concatenate(([x[-1]], new_x))
        self.y_exp = np.concatenate(([y[-1]], new_y))

        self.popt_exp = popt
        self.is_linear = is_linear

    @staticmethod
    def _linear_fit(x, y):
        m = cp.Variable()
        c = cp.Variable()

        constraints = [y[i] <= m * x[i] + c for i in range(len(x))]
        mean_y = sum(y) / len(y)
        tss = cp.sum_squares(y - mean_y)
        rss = cp.sum_squares(y - (m * x + c))
        r_squared = 1 - rss / tss
        objective = cp.Maximize(r_squared)
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
        except:
            return None, None, None

        if not r_squared.value:
            r_squared_lin = None
        else:
            r_squared_lin = r_squared.value

        return m.value, c.value, r_squared_lin

    @staticmethod
    def _logarithmic_fit(x, y):
        m = cp.Variable()
        c = cp.Variable()

        constraints = [y[i] <= m * cp.log(x[i]) + c for i in range(len(x))]
        mean_y = sum(y) / len(y)
        tss = cp.sum_squares(y - mean_y)
        rss = cp.sum_squares(y - (m * cp.log(np.array(x)) + c))
        r_squared = 1 - rss / tss
        objective = cp.Maximize(r_squared)
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
        except:
            return None, None, None

        if not r_squared.value:
            r_squared_log = None
        else:
            r_squared_log = r_squared.value

        return m.value, c.value, r_squared_log

    def curve_fit_linear(self):

        if len(self.y_exp) == 2:

            y_max_index = np.argmax(self.y_exp)
            x_max_index = np.argmax(self.x_exp)
            x = [self.x_exp[0], self.x_exp[y_max_index]]
            y = [self.y_exp[0], self.y_exp[y_max_index]]

            if self.x_exp[x_max_index] < 0.09:

                target_y = self.y_exp[x_max_index] * 1.1
                m_fit = (abs(target_y - self.y_exp[x_max_index]) / abs(1 - self.x_exp[x_max_index]))
                c_fit = self.y_exp[x_max_index] - m_fit * self.x_exp[x_max_index]

                popt = [float(m_fit), float(c_fit)]
            else:

                m_linear, c_linear, r_squared_linear = self._linear_fit(x, y)
                m_logarithmic, c_logarithmic, r_squared_logarithmic = self._logarithmic_fit(x, y)

                if r_squared_linear and r_squared_logarithmic:
                    if r_squared_linear >= r_squared_logarithmic:
                        popt = [m_linear, c_linear]
                    else:
                        popt = [m_logarithmic, c_logarithmic]
                        self.is_logarithmic = True
                elif r_squared_linear:
                    popt = [m_linear, c_linear]
                elif r_squared_logarithmic:
                    popt = [m_logarithmic, c_logarithmic]
                    self.is_logarithmic = True
                else:
                    popt = [None, None]

                if not popt[0] and not popt[1]:
                    self.is_logarithmic = False

                    popt, pcov = optimize.curve_fit(self._linear, x, y)

            self.popt_linear = [float(popt[0]), float(popt[1])]
            self.intersection = float(self.x_exp[0])

            x_fit, x_fit_2, y_fit, y_fit_2 = self._arrange_graph_variables()
            #self._create_graph(x_fit, y_fit, x_fit_2, y_fit_2)

        elif len(self.y_exp) > 2:

            max_index = np.argmax(self.x_exp)
            if self.x_exp[max_index] < 0.09:

                target_y = self.y_exp[max_index] * 1.1
                m_fit = (abs(target_y - self.y_exp[max_index]) / abs(1 - self.x_exp[max_index]))
                c_fit = self.y_exp[max_index] - m_fit * self.x_exp[max_index]

            else:
                m_linear, c_linear, r_squared_linear = self._linear_fit(self.x_exp, self.y_exp)
                m_logarithmic, c_logarithmic, r_squared_logarithmic = self._logarithmic_fit(self.x_exp, self.y_exp)

                if r_squared_linear and r_squared_logarithmic:
                    if r_squared_linear >= r_squared_logarithmic:
                        m_fit = m_linear
                        c_fit = c_linear
                    else:
                        m_fit = m_logarithmic
                        c_fit = c_logarithmic
                        self.is_logarithmic = True
                elif r_squared_linear:
                    m_fit = m_linear
                    c_fit = c_linear
                elif r_squared_logarithmic:
                    m_fit = m_logarithmic
                    c_fit = c_logarithmic
                    self.is_logarithmic = True
                else:
                    m_fit = None
                    c_fit = None

                if not m_fit and not c_fit and m_fit != 0 and c_fit != 0:

                    self.is_logarithmic = False

                    ar_x = np.array(self.x_exp)
                    ar_y = np.array(self.y_exp)
                    # Sort the data points by ar_x-values
                    sorted_indices = np.argsort(ar_x)
                    ar_x_sorted = ar_x[sorted_indices]
                    ar_y_sorted = ar_y[sorted_indices]

                    best_fit_r_squared = -999
                    m_fit = None
                    c_fit = None

                    for i in range(len(ar_x_sorted) - 1):
                        for j in range(i + 1, len(ar_x_sorted)):
                            ar_x1, ar_x2 = ar_x_sorted[i], ar_x_sorted[j]
                            ar_y1, ar_y2 = ar_y_sorted[i], ar_y_sorted[j]

                            m = (ar_y2 - ar_y1) / (ar_x2 - ar_x1)
                            b = ar_y1 - m * ar_x1

                            line = m * ar_x + b

                            if np.all(np.round(ar_y, 0) <= np.round(line, 0)):
                                fitted_y_ar = self._linear(ar_x_sorted, m, b)
                                r_squared = self._r_squared(ar_y_sorted, fitted_y_ar)

                                if r_squared > best_fit_r_squared:
                                    best_fit_r_squared = r_squared
                                    m_fit = m
                                    c_fit = b

            self.popt_linear = [float(m_fit), float(c_fit)]
            self.intersection = float(self.x_exp[0])

            x_fit, x_fit_2, y_fit, y_fit_2 = self._arrange_graph_variables()
            #self._create_graph(x_fit, y_fit, x_fit_2, y_fit_2)

        else:

            target_y = self.y_exp * 1.1
            a = (abs(target_y - self.y_exp) / abs(1 - self.x_exp)) if abs(1 - self.x_exp) != 0 else 0
            b = self.y_exp - a * self.x_exp

            self.popt_linear = [float(a), float(b)]
            self.intersection = float(self.x_exp)

            x_fit, x_fit_2, y_fit, y_fit_2 = self._arrange_graph_variables()
            #self._create_graph(x_fit, y_fit, x_fit_2, y_fit_2)

    def _arrange_graph_variables(self):
        # Fit exponencial
        x_fit_2 = np.linspace(0, self.x_exp[0], 100)

        if not self.is_linear:
            y_fit_2 = self._exp(x_fit_2, *self.popt_exp)
        else:
            y_fit_2 = self._exception_linear(x_fit_2, *self.popt_exp)

        # Fit linear/logaritmico
        x_fit = np.linspace(self.x_exp[0], self.x_exp[-1], 100)

        if self.is_logarithmic:
            y_fit = self._logarithmic(x_fit, *self.popt_linear)
        else:
            y_fit = self._linear(x_fit, *self.popt_linear)

        return x_fit, x_fit_2, y_fit, y_fit_2

    def _create_graph(self, x_fit, y_fit, x_fit_2, y_fit_2):
        plt.figure()
        plt.scatter(self.x_value, self.y_value, label='Dados')
        if self.is_logarithmic:
            plt.plot(x_fit, y_fit, 'g', label='Log')
        else:
            plt.plot(x_fit, y_fit, 'r', label='Reta')

        plt.plot(x_fit_2, y_fit_2, '--b', label='Exponencial')
        plt.ylim(bottom=0)
        plt.xlim(left=0)
        plt.xlim(right=self.x_exp[-1])
        plt.legend()
        plt.grid(True)
        plt.title(self.graph_title)
        if self.show_graph:
            plt.show()
        plt.clf()

    def get_important_variables(self):
        return self.popt_exp, self.intersection, self.popt_linear, self.is_linear, self.is_logarithmic
