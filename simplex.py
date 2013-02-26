"""
Plot 3D points onto 2D triangle (simplex).

Adapted from work by David Andrzejewski (david.andrzej@gmail.com)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.ticker as ticker


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


def plotSimplex(topic1, topic2, topic3, grayscale=True):
    """
    Plot 3D data onto 2D triangular simplex.

    Parameters
    ----------
    topic1 : (N, 3) array_like
    topic2 : (N, 3) array_like
    topic3 : (N, 3) array_like
    grayscale : bool
        Whether to show plot in grayscale (True) or color (False).

    """

    # test data
    simplex1 = projectSimplex(topic1)
    simplex2 = projectSimplex(topic2)
    simplex3 = projectSimplex(topic3)

    # plot
    plt.figure()
    ax = plt.subplot(111)

    # plot data
    kwargs = {'s': 70}

    if grayscale is False:
        colors = ['#dc322f', '#000000', '#268bd2']
    else:
        colors = ['#000000', '#000000', '#999999']

    ax.scatter(simplex1[:, 0], simplex1[:, 1], label='Topic 1',
               marker='o', color=colors[0], **kwargs)
    ax.scatter(simplex2[:, 0], simplex2[:, 1], label='Topic 2',
               marker='x', color=colors[1], **kwargs)
    ax.scatter(simplex3[:, 0], simplex3[:, 1], label='Topic 3',
               marker='^', color=colors[2], **kwargs)

    # add legend
    ax.legend()

    # add triangle outline
    lines_triangle = lines.Line2D([0.0, 0.5, 1.0, 0.0],
                                  [0.0, np.sqrt(3) / 2, 0.0, 0.0],
                                  color='k')
    ax.add_line(lines_triangle)

    # add labels to triange vertex
    text_kwargs = {'fontsize': 14}
    ax.text(-0.05, -0.05, r'$\theta_1$', **text_kwargs)
    ax.text(1.05, -0.05, r'$\theta_2$', **text_kwargs)
    ax.text(0.50, np.sqrt(3) / 2 + 0.05, r'$\theta_3$', **text_kwargs)

    # remove axes labels
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.NullLocator())

    # set limits
    ax.set_xlim([-0.2, 1.2])
    ax.set_ylim([-0.2, 1.2])

    plt.show()


#
# Demo simplex plotting
#

# create demo data (Nx3 arrays)
topic1 = np.array([[0.9, 0.0, 0.0],
                   [0.6, 0.2, 0.1],
                   [0.7, 0.1, 0.2]])
topic2 = np.array([[0.2, 0.9, 0.1],
                   [0.0, 0.7, 0.0],
                   [0.1, 0.8, 0.2]])
topic3 = np.array([[0.1, 0.2, 0.8],
                   [0.0, 0.0, 0.6],
                   [0.2, 0.1, 0.9]])

plotSimplex(topic1, topic2, topic3, grayscale=False)
