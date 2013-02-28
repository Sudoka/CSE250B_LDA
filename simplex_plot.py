"""
Plot 3D points onto 2D triangle (simplex).

Based on work by David Andrzejewski (david.andrzej@gmail.com)
source: https://gist.github.com/davidandrzej/939840
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines


def projectSimplex(data):
    """
    Projects 3D data onto 2D triangular simplex.

    Parameters
    ----------
    data : (N, 3) array_like
        3D data (assumed to be normalized from 0.0 to 1.0).

    Returns
    -------
    points : (N, 2) array_like
        3D data projected onto 2D triangular simplex.

    """

    points = np.zeros((data.shape[0], 2))

    data = np.asarray(data)

    for i in range(data.shape[0]):

        # initialize to triangle centroid
        x = 1.0 / 2
        y = 1.0 / (2 * np.sqrt(3))

        # vector 1: bisect out of lower left vertex
        p1 = data[i, 0]
        x = x - (1.0 / np.sqrt(3) * p1 * np.cos(np.pi / 6))
        y = y - (1.0 / np.sqrt(3) * p1 * np.sin(np.pi / 6))

        # vector 2: bisect out of lower right vertex
        p2 = data[i, 1]
        x = x + (1.0 / np.sqrt(3) * p2 * np.cos(np.pi / 6))
        y = y - (1.0 / np.sqrt(3) * p2 * np.sin(np.pi / 6))

        # vector 3: bisect out of top vertex
        p3 = data[i, 2]
        y = y + (1.0 / np.sqrt(3) * p3)

        points[i, :] = (x, y)

    return points


def plotSimplex(thetas, dataset, grayscale=True, save=False, show=True):
    """
    Plot 3D data onto 2D triangular simplex.

    Parameters
    ----------
    thetas : (N, 3) array_like
    dataset : str
        Which dataset are the theta values calculated from.
    grayscale : bool
        Whether to show plot in grayscale (True) or color (False).

    """

    plt.figure(facecolor='w')
    ax = plt.subplot(111)

    # add triangle outline
    lines_triangle = lines.Line2D([0.0, 0.5, 1.0, 0.0],
                                  [0.0, np.sqrt(3) / 2, 0.0, 0.0],
                                  color='k')
    ax.add_line(lines_triangle)

    # add labels to triange vertex
    text_kwargs = {'fontsize': 24}
    ax.text(-0.10, -0.10, r'$\theta_1$', **text_kwargs)
    ax.text(1.05, -0.10, r'$\theta_2$', **text_kwargs)
    ax.text(0.50, np.sqrt(3) / 2 + 0.10, r'$\theta_3$', **text_kwargs)

    # remove x-y axis
    ax.axis('off')

    # set limits
    ax.set_xlim([-0.2, 1.2])
    ax.set_ylim([-0.2, 1.2])

    # plot formatting
    kwargs = {'s': 120}
    markers = ['o', 's', '^']
    if grayscale is False:
        colors = ['#268bd2', '#000000', '#dc322f']
    else:
        colors = ['w', 'k', '#999999']

    # plot data
    if dataset is 'classic400':
        # split data by topic
        topic1 = thetas[0:99, :]
        topic2 = thetas[100:199, :]
        topic3 = thetas[200:, :]

        # convert data to simplex coordinates
        simplex1 = projectSimplex(topic1)
        simplex2 = projectSimplex(topic2)
        simplex3 = projectSimplex(topic3)

        # plot data onto simplex
        ax.scatter(simplex1[:, 0], simplex1[:, 1], label='Topic 1',
                   marker=markers[0], color=colors[0], edgecolor='k', **kwargs)
        ax.scatter(simplex2[:, 0], simplex2[:, 1], label='Topic 2',
                   marker=markers[1], color=colors[1], edgecolor='k', **kwargs)
        ax.scatter(simplex3[:, 0], simplex3[:, 1], label='Topic 3',
                   marker=markers[2], color=colors[2], edgecolor='k', **kwargs)

        plot_name = 'classic400_thetas.pdf'

    elif dataset is 'KOS':
        # convert data to simplex coordinates
        simplex1 = projectSimplex(thetas)

        # plot data onto simplex
        ax.scatter(simplex1[:, 0], simplex1[:, 1], label='Topic 1',
                   marker='D', color='#888888', edgecolor='k', **kwargs)

        plot_name = 'kos_thetas.pdf'

    else:
        print "Error: incorrect dataset given."

    if save is True:
        plt.savefig(plot_name)

    if show is True:
        plt.show()


#
# Classic400
#
thetas_classic400 = np.genfromtxt('classic400/classic400_thetas.csv',
                                  delimiter=',')

# convert from log-10
thetas_classic400 = np.exp(thetas_classic400)

plotSimplex(thetas_classic400, 'classic400', grayscale=True, save=True)


#
# KOS
#
thetas_kos = np.genfromtxt('KOS400/KOS400_thetas.csv', delimiter=',')

# convert from log-10
thetas_kos = np.exp(thetas_kos)

plotSimplex(thetas_kos, 'KOS', grayscale=True, save=True)
