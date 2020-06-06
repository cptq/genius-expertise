""" Our settings for publication plots. """
import matplotlib.pyplot as plt

def two_col():
    """ For when there are two figures in one column. """
    plt.rcParams.update({'font.size': 7})
    plt.rcParams['lines.markeredgecolor'] = 'k'
    plt.rcParams['lines.markeredgewidth'] = .2
    plt.rcParams['lines.markersize'] = 2
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams.update({'figure.figsize':[1.45,1.2]})
    plt.rcParams.update({'font.family':'serif'})
    plt.rcParams.update({'figure.dpi':300})

def one_col():
    """ For when there is one figure in a column. """
    plt.rcParams.update({'font.size': 7})
    plt.rcParams['lines.markeredgecolor'] = 'k'
    plt.rcParams['lines.markeredgewidth'] = .2
    plt.rcParams['lines.markersize'] = 2
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams.update({'figure.figsize':[2.9,1.2]})
    plt.rcParams.update({'font.family':'serif'})
    plt.rcParams.update({'figure.dpi':300})
