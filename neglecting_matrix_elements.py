import numpy as np
import matplotlib.pyplot as plt
import time

from telecom import get_ising_parameters, binary_to_decimal, probability_histogram
from annealing import solve_annealing, modify_coupling_matrix, get_groundstate, get_state, find_state_index, draw_graph, matrix_histogram, is_connected, adjacency_from_couplings, degree_of_nodes
from hamming_distance_distribution import hamming_distance


def main():
    # -------- Parameters -------
    N = 5               # number of users
    M = 4               # length of id-sequence for every user
    K = 100              # number of antennas
    xi = 0              # std of thermal noise
    
    which_ctl_fct = 0   # 0: linear control function, 1: optimal control function
    neglection_rule = 1 # 0: neglect only smallest matrix element, 1: neglect everything below thresold
    neglection_thres = 0.1  # thres only used if reglection_rule == 1

    nb_pts_gap = 20     # number of points for the gap computation
    nb_pts_time = 30    # number of points for resolution of the time dependant Schrodinger's equation
    epsilon = 0.1       # precision level for the control function (valid for both, linear and optimal scheduling)
    gamma = 1           # strength of the transverse field, irrelevant for us 

    resol = 100
    height_plt = 6
    width_plt = height_plt

    show = [0, 3, 4] # 0:visualization of J, 1: energy levels and gap, 2: control function, 3: evolution of the overlap, 4: probabilities in the final state

    # ------- Program ----------
    
    # activity pattern
    alpha = np.zeros(N)
    alpha[0] = 1
    alpha[1] = 1
    alpha_decimal = binary_to_decimal(alpha)
    psi_alpha = get_state(alpha)

    # Problem instance with couplings
    t0 = time.time()
    J, b, *_ = get_ising_parameters(N, M, alpha, K, xi)

    # modification of coupling matrix (neglecting matrix element)
    t1 = time.time()
    J_n, where_n = modify_coupling_matrix(J, neglection_rule, neglection_thres)
    

    # solve the annealing process
    proba_coef, sigma_z_exp, basis, times_tab, tp, up, spectrum_tab, squared_gap, Hscheduled = solve_annealing(J, b, gamma, epsilon, which_ctl_fct, nb_pts_gap, nb_pts_time)
    t2 = time.time()
    proba_coef1, sigma_z_exp1, basis1, times_tab1, tp1, up1, spectrum_tab1, squared_gap1, Hscheduled1 = solve_annealing(J_n, b, gamma, epsilon, which_ctl_fct, nb_pts_gap, nb_pts_time)
    t3 = time.time()
    print(f"time for the annealing computations: {t2-t1:.3f} s, {t3-t2:.3f} s")
    print(f"time for total solving of problem: {t2-t0:.3f} s")

    # find the index of the state which corresponds to the activity pattern
    i_alpha = find_state_index(psi_alpha, basis)
    i_alpha1 = find_state_index(psi_alpha, basis1)

    # Check if Graph with neglected matrix elements is connected
    A = adjacency_from_couplings(J_n)
    print(f"Is the graph connected? -> {is_connected(A)}")
    degrees = degree_of_nodes(A)
    print(f"The nodes have the degrees {degrees}")
    print(f"degrees: min: {np.min(degrees)} avg: {np.mean(degrees)}, max: {np.max(degrees)}")

    # ---------- plotting ------------
    # visualization of the matrix and the neglected matrix element
    if 0 in show:
        N_row = 1
        N_col = 2
        fig = plt.figure(figsize=(N_col*width_plt, N_row*height_plt))
        matrix_histogram(fig, -J, where_n, N_row, N_col, 1)

        ax2 = fig.add_subplot(N_row, N_col, 2)
        draw_graph(ax2, J_n)
        plt.tight_layout()
        plt.show()

    # energy levels and gap
    if 1 in show:
        N_row = 1
        N_col = 2
        fig, ax = plt.subplots(N_row, N_col, figsize=(N_col*width_plt, N_row*height_plt))#, dpi=resol)

        ax[0].plot(up, spectrum_tab[:,0], marker='o', label="$\\varepsilon_0$", color="blue")
        ax[0].plot(up, spectrum_tab[:,1], marker='o', label="$\\varepsilon_1$", color="green")
        ax[0].plot(up, spectrum_tab[:,2], marker='o', label="$\\varepsilon_2$", color="red")

        ax[0].plot(up, spectrum_tab1[:,0], marker='o', label="$\\varepsilon_0$", color="blue", ls="dashed")
        ax[0].plot(up, spectrum_tab1[:,1], marker='o', label="$\\varepsilon_1$", color="green", ls="dashed")
        ax[0].plot(up, spectrum_tab1[:,2], marker='o', label="$\\varepsilon_2$", color="red", ls="dashed")
        
        ax[0].set_xlabel("Control function $u(t)$")
        ax[0].set_ylabel("Eigenvalues")
        ax[0].invert_xaxis()
        ax[0].legend()
        ax[0].grid()

        ax[1].plot(up, squared_gap, marker='o', label="$\\Delta^2(u)$", color="black")
        ax[1].plot(up1, squared_gap1, marker='o', label="$\\Delta^2(u)$", color="black", ls="dashed")
        ax[1].set_xlabel("Control function $u(t)$")
        ax[1].set_ylabel("Squared gap $\\Delta^2$")
        ax[1].invert_xaxis()
        ax[1].legend()
        ax[1].grid()

        plt.tight_layout()
        plt.show()


    # control function
    if 2 in show:
        plt.figure(figsize=(width_plt, height_plt))
        plt.plot(tp, up, marker='o', color="royalblue")
        plt.plot(tp1, up1, marker='o', color="royalblue", ls="dashed")
        plt.xlabel("Time")
        plt.ylabel("Control function $u(t)$")
        plt.grid()
        plt.show()

    # Evolution of the overlap
    if 3 in show:

        plt.figure(figsize=(width_plt, height_plt), dpi=resol)
        plt.plot(times_tab, proba_coef[i_alpha], marker="D", label=r"$\langle\psi|\alpha\rangle$", color="lime")
        plt.plot(times_tab1, proba_coef[i_alpha1], marker="D", label=r"$\langle\psi_n|\alpha\rangle$", color="lime", ls="dashed")
        plt.plot(times_tab, proba_coef[0], marker="D", label=r"$\langle\psi|\psi_{0}\rangle$", color="black")
        plt.plot(times_tab1, proba_coef1[0], marker="D", label=r"$\langle\psi_n|\psi_{0, n}\rangle$", color="black", ls="dashed")
        plt.xlabel("Time")
        plt.ylabel("Overlap with the solution")
        plt.ylim(0,1)
        plt.legend(loc="upper left")
        plt.grid()
        plt.show()

    # amplitudes in the final state
    if 4 in show:
        xlabels, coef_to_plot = probability_histogram(basis, proba_coef, order_by="binary")
        xlabels1, coef_to_plot1 = probability_histogram(basis1, proba_coef1, order_by="binary")

        # verify that the ordering is the same
        print("verification of the order ")
        for i in range(len(xlabels)):
            if xlabels[i] != xlabels1[i]:
                print("!!! ORDER OF THE STATES IS NOT THE SAME!!! -> Histogram not correct!")

        #get the GS of the final Hamiltonian
        print(Hscheduled(1).shape)
        _, gs_string, gs_array = get_groundstate(Hscheduled, 1)
        print("Ground state of the final Hamiltonian:", gs_string)

        _, gs_string1, gs_array1 = get_groundstate(Hscheduled1, 1)
        print("Ground state of the final Hamiltonian with neglected matrix elements:", gs_string1)
        print("The initial activity pattern is" , alpha)
        print(f"Bit error ratio between both annealing results: {hamming_distance(gs_array, gs_array1)/N}")


        plt.figure(figsize=(15, 5))
        plt.bar(xlabels, coef_to_plot, color="royalblue", edgecolor="royalblue")
        plt.bar(xlabels, coef_to_plot1, color="red", edgecolor="red", fill=False)
        plt.plot([alpha_decimal, alpha_decimal], [1, 1], color="magenta", marker="o")
        plt.xticks(rotation=90)
        plt.xlabel("States")
        plt.ylabel("Probability")
        #plt.yscale("log")
        plt.grid()
        plt.show()

        print(np.sum(coef_to_plot))
    

if __name__ == "__main__":
    main()