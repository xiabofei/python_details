# encoding=utf8

import numpy as np
import matplotlib.pyplot as plt
from ipdb import set_trace as st


class ValleyLoss(object):
    __doc__ = 'define a valley look like loss function y = x1*x1 + 50*x2*x2'

    valley_rate = 50

    @classmethod
    def f(cls, variable):
        """
        'f' for loss function
        :param variable:
        :return:
        """
        return variable[0] * variable[0] + cls.valley_rate * variable[1] * variable[1]

    @classmethod
    def g(cls, variable):
        """
        'g' for gradient of loss function
        :param variable:
        :return:
        """
        # return np.array([2 * variable[0], 2 * cls.valley_rate * variable[1]])
        return np.array([-2 * variable[0], 2 * variable[1]])

    @classmethod
    def plot_loss_function_contour_fig(cls, arrs=None):
        colors = ['blue', 'red', 'black']
        X, Y = np.meshgrid(np.linspace(-8, 8, 1000), np.linspace(-8, 8, 1000))
        Z = X * X + cls.valley_rate * Y * Y
        plt.figure(figsize=(9, 6))
        plt.contour(X, Y, Z, colors='black')
        plt.plot(0, 0, marker='*')
        if arrs is not None:
            for index, arr in enumerate(arrs):
                for i in range(len(arr) - 1):
                    plt.plot(arr[i:i + 2, 0], arr[i:i + 2, 1], color=colors[index])


class Optimizer(object):
    ZERO = 1e-6

    @classmethod
    def gd(cls, start_point, lr=0.019, iter_rounds=50):
        x = np.array(start_point, dtype='float64')
        x_trace = [x.copy()]
        for i in range(iter_rounds):
            grad = ValleyLoss.g(x)
            x += -lr * grad
            x_trace.append(x.copy())
            print '[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x)
            if abs(sum(grad)) < cls.ZERO:
                break
        return np.array(x_trace)

    @classmethod
    def gd_with_momentum(cls, start_point, lr=0.019, iter_rounds=50, mu=0.7):
        x = np.array(start_point, dtype='float64')
        x_trace = [x.copy()]
        v = 0
        for i in range(iter_rounds):
            grad = ValleyLoss.g(x)
            v = mu * v - lr * grad
            x += v
            x_trace.append(x.copy())
            print '[ Epoch {0} ] grad = {1}, v = {2}, x = {3}'.format(i, grad, v, x)
            if abs(sum(grad)) < cls.ZERO:
                break
        return np.array(x_trace)

    @classmethod
    def nesterov(cls, start_point, lr=0.019, iter_rounds=50, mu=0.7):
        x = np.array(start_point, dtype='float64')
        x_ahead_trace = [x.copy()]
        x_trace = [x.copy()]
        v = 0
        for i in range(iter_rounds):
            x_ahead = x + mu * v
            x_ahead_trace.append(x_ahead.copy())
            grad = ValleyLoss.g(x_ahead)
            v = mu * v - lr * grad
            x += v
            x_trace.append(x.copy())
            print '[ Epoch {0} ] x_ahead = {1}, grad = {2}, v = {3},  x = {4}'.format(i, x_ahead, grad, v, x)
            if abs(sum(grad)) < cls.ZERO:
                break
        return np.array(x_trace), np.array(x_ahead_trace)

    @classmethod
    def adagrad(cls, start_point, iter_rounds=50, eta=0.01, epsilon=1e-8):
        x = np.array(start_point, dtype='float64')
        x_trace = [x.copy()]
        g_norm2 = np.zeros_like(x)
        # st(context=21)
        for i in range(iter_rounds):
            grad = ValleyLoss.g(x)
            g_norm2 += grad * grad
            v = -(eta / np.sqrt(g_norm2 + epsilon)) * grad
            x += v
            x_trace.append(x.copy())
            print '[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x)
            if abs(sum(grad)) < cls.ZERO:
                break
        return np.array(x_trace)

    @classmethod
    def adadelta(cls, start_point, iter_rounds=50, epsilon=0.1, mu=0.5):
        x = np.array(start_point, dtype='float64')
        x_trace = [x.copy()]
        G = np.zeros_like(x)
        v_accumulate = np.zeros_like(x)
        for i in range(iter_rounds):
            grad = ValleyLoss.g(x)
            G = mu * G + (1 - mu) * (grad ** 2)
            v = - (np.sqrt(v_accumulate + epsilon) / np.sqrt(G + epsilon))
            v_accumulate = mu * v_accumulate + (1-mu) * (v**2)
            x += v
            x_trace.append(x.copy())
            print '[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x)
            if abs(sum(grad)) < cls.ZERO:
                break
        return np.array(x_trace)

    @classmethod
    def rmsprop(cls, start_point, iter_rounds=50, eta=0.01, epsilon=1e-8):
        x = np.array(start_point, dtype='float64')
        x_trace = [x.copy()]
        g_expectation = np.zeros_like(x)
        for i in range(iter_rounds):
            grad = ValleyLoss.g(x)
            g_expectation = 0.9 * g_expectation + 0.1 * grad ** 2
            v = - (eta / np.sqrt(g_expectation + epsilon)) * grad
            x += v
            x_trace.append(x.copy())
            print '[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x)
            if abs(sum(grad)) < cls.ZERO:
                break
        return np.array(x_trace)

    @classmethod
    def adam(cls, start_point, iter_rounds=50, eta=0.01, epsilon=1e-8, beta1=0.7, beta2=0.9):
        x = np.array(start_point, dtype='float64')
        x_trace = [x.copy()]
        g_expectation = np.zeros_like(x)
        m = np.zeros_like(x)
        for i in range(iter_rounds):
            grad = ValleyLoss.g(x)
            m = beta1 * m + (1 - beta1) * grad
            g_expectation = beta2 * g_expectation + (1 - beta2) * grad ** 2
            v = - (eta * (m / (1 - beta1 ** (i + 1))) / np.sqrt((g_expectation / (1 - beta2 ** (i + 1))) + epsilon))
            x += v
            x_trace.append(x.copy())
            print '[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x)
            if abs(sum(grad)) < cls.ZERO:
                break
        return np.array(x_trace)


x_trace_arr = []

# z = x² + 50y² : convergence speed case
# x_trace_arr.append(Optimizer.gd([-150, 75], 1e-2*1.6, 30))
# x_trace_arr.append(Optimizer.gd_with_momentum([150, 75], 1e-2 * 1.6, 20, 0.7))
# x_trace, x_ahead_trace = Optimizer.nesterov([-150, 75], 1e-2 * 1.2, 10, 0.7)
# x_trace_arr.append(x_trace)
# x_trace_arr.append(x_ahead_trace)

# z = -x² + y² : saddle point case
print('SGD')
x_trace_arr.append(Optimizer.gd([-0.001, 4], 0.1, 25))
print('sgd+momentum')
x_trace_arr.append(Optimizer.gd_with_momentum([-0.001, 4], 0.1, 25, 0.9))
print('adaGrad')
x_trace_arr.append(Optimizer.adagrad([-0.001, 4], 20, 0.5))
# print('adaDelta')
# x_trace_arr.append(Optimizer.adadelta([-0.001, 4], 20))
# print('rmsprop')
# x_trace_arr.append(Optimizer.rmsprop([150, 75], 5, 1e-8))
# print('adam')
# x_trace_arr.append(Optimizer.adam([-150, 75], 5, 10, 1e-8))
ValleyLoss.plot_loss_function_contour_fig(x_trace_arr)
plt.show()
