"""useful functions for drawing and analyzing graphs represented by a matrix"""

import numpy as np

def matrix_histogram(fig, matrix, where_red, N_row=1, N_col=1, ax_index=1):
    """draws the matrix elements as a histogram
    Parameters:
    - fig           plt.figure() to draw on
    - matrix        matrix whose elements should be represented by a 2d histogram
    - where_red     np.where(...) that contains the indices of those matrix elements that should be highlighted in red
    - N_row         the number of rows of the figure (necessary if the histogram is not the only plot in the figure)
    - N_col         the number of columns of the figure (necessary if the histogram is not the only plot in the figure)
    - ax_index      index of the matrix histogram plot in the figure (necessary if the histogram is not the only plot in the figure)
    """
    N = np.shape(matrix)[0]

    x_pos, y_pos = np.meshgrid(np.arange(N), np.arange(N))
    x_pos = x_pos.flatten()
    y_pos = y_pos.flatten()
    z_pos = np.zeros_like(x_pos)
    height = matrix.flatten()

    # set list of colors such that the by where_red selected ones are red
    colors = ["skyblue"]*len(height)
    for i in range(len(where_red[0])):
        idx_n = where_red[0][i]*N + where_red[1][i]  # where is ([i], [j]) for the index (i, j), so it is necessary to fetch element 0 in order to get only the number and not the list
        colors[idx_n] = "red"

    # generate plot
    ax = fig.add_subplot(N_row, N_col, ax_index, projection="3d")
    ax.bar3d(x_pos, y_pos, z_pos, dx=0.8, dy=0.8, dz=height, color=colors, edgecolor="grey")
    ax.set_xlabel(r"$j$")
    ax.set_ylabel(r"$i$")
    ax.set_zlabel(r"$-J_{ij}$")
    ax.invert_yaxis()       # to be able to look at the matrix as if it was "lying on the table"
    ax.view_init(elev=40, azim=-110, roll=0)

    return ax


def draw_graph(ax, matrix, nodecolor = "blue", edgecolor="green"):
    """Draws a graph from matrix. Uses only upper right of the matrix and assumes an undirected graph.
    Parameters:
    - ax        axes to draw the graph on 
    - J_n       matrix containing the couplings between the nodes
    - nodecolor
    - edgecolor
    """

    N = np.shape(matrix)[0]
    r = 1
    phi = np.linspace(0, 2*np.pi*(1-1/N), N)
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    ax.scatter(x, y, color=nodecolor)
    for i in range(N):
        ax.annotate(f"{i}", xy=(x[i], y[i]), color=nodecolor, xytext=(1.1*x[i], 1.1*y[i]))
        for j in range(i+1, N):     # iterate over columns of J
            ax.plot([x[i], x[j]], [y[i], y[j]], color=edgecolor, linewidth=10*np.abs(matrix[i, j]))
    ax.set_aspect("equal")
    ax.set_xlim(1.1*np.array([-r, r]))
    ax.set_ylim(1.1*np.array([-r, r]))


def is_connected(A, verbose=False, return_B=False):
    """determines whether an undirected graph is connected or not by computing sum of successive powers of the adjacency matrix
    Parameters:
    - A         NxN adjacency matrix of the graph (must be symmetric for undirected graph)
    - return_B  if matrix B should be returned

    Returns:
    - connected False = not connected, True = connected
    - B         B_ij is number of walks from node i to j with length < N 
    """
    N = np.shape(A)[0]
    B = np.zeros_like(A)
    C = np.identity(N, dtype=int)
    for _ in range(N-1):
        C = C@A
        B += C

    if verbose:
        print(f"B = \n{B}")

    if len(np.where(B==0)[0]) == 0:
        connected = True
    else:
        connected = False
    
    if return_B:
        return connected, B
    else:
        return connected


def degree_of_nodes(A):
    """Determines the degree of the nodes from the adjacency matrix"""
    return np.sum(A, axis=0)