'''
Plotters

shorthand plotting funcs
'''

from matplotlib import pyplot as plt

def line_onetwod(y, **kwargs):

    x = kwargs.get('x', None)
    x_label = kwargs.get('x_label', '')
    y_label = kwargs.get('y_label', '')

    if x:
        plt.plot(y, x)
        plt.xlabel(x_label)

    else:
        plt.plot(y)

    plt.ylabel(y_label)
    plt.show()

def scatter_2d(x, y, **kwargs):

    x_label = kwargs.get('x_label', '')
    y_label = kwargs.get('y_label', '')

    plt.scatter(y, x)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

