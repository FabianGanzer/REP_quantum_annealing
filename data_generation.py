import numpy as np
import time
from annealing import solve_annealing, modify_coupling_matrix, get_groundstate
from telecom import get_ising_parameters


def warn_user(path):
    print("ATTENTION: append_to_existing_data is set to False!!!")
    print(f"The newly generated data will NOT be appended to existing files, instead, NEW files in {path} will be generated! ")
    N_ask_user = 10
    for i in range(N_ask_user):
        print(f"{N_ask_user-i} Are you sure? You have 10s to abort ...")
        time.sleep(1)


def check_data(fname):
    loaded_data = np.load(fname, allow_pickle=True).item()
    print("-----------------------------------------------------------")
    print(f"Check of data file {fname}")
    print(f"size                = {loaded_data['size']}")
    print(f"np.shape(J)         = {np.shape(loaded_data['J'])}")
    print(f"np.shape(J_n)       = {np.shape(loaded_data['J_n'])}")
    print(f"len(where_n)        = {len(loaded_data['where_n'])}")
    print(f"np.shape(gs_array)  = {np.shape(loaded_data['gs_array'])}")
    print(f"np.shape(alpha)     = {np.shape(loaded_data['alpha'])}")
    print(f"neglection_thres    = {loaded_data['neglection_thres']}")
    print(f"N                   = {loaded_data['N']}")
    print(f"M                   = {loaded_data['M']}")
    print(f"K                   = {loaded_data['K']}")
    print(f"xi                  = {loaded_data['xi']}")
    print(f"epsilon             = {loaded_data['epsilon']}")
    print(f"gamma               = {loaded_data['gamma']}")
    print(f"which_ctl_fct       = {loaded_data['which_ctl_fct']}")
    print("-----------------------------------------------------------")



def generate_data_for_one_thres(neglection_thres, N_per_thres, N, M, alpha, K, xi, gamma, epsilon, which_ctl_fct, nb_pts_gap, nb_pts_time):
    """simulates the annealing for a single given neglection thresold"""
    # define data collection arrays
    J_data              = np.zeros(shape=(N_per_thres, N, N))
    J_n_data            = np.zeros_like(J_data)
    alpha_data          = np.zeros(shape=(N_per_thres, N), dtype=int)
    gs_array_data       = np.zeros(shape=(N_per_thres, N), dtype=int)
    where_n_data        = []

    # simulate the annealing processes
    for j in range(N_per_thres):
        # solve problem
        J, b, *_ = get_ising_parameters(N, M, alpha, K, xi, False)
        J_n, where_n = modify_coupling_matrix(J, 1, neglection_thres, False)
        _, _, _, _, _, _, _, _, Hscheduled = solve_annealing(J, b, gamma, epsilon, which_ctl_fct, nb_pts_gap, nb_pts_time, False)
        _, _, gs_array = get_groundstate(Hscheduled, 1)

        # fill arrays
        J_data[j]           = J
        J_n_data[j]         = J_n
        gs_array_data[j]    = gs_array
        alpha_data[j]       = alpha
        where_n_data.append(where_n)


    return J_data, J_n_data, alpha_data, gs_array_data, where_n_data


def generate_data_and_save():
    # ------------ Parameters -------------
    N_per_thres = 20
    thres_min = 0.0
    thres_max = 0.4
    thres_step = 0.01

    N = 5               # number of users
    M = 4               # length of id-sequence for every user
    K = 100              # number of antennas
    xi = 0              # std of thermal noise

    which_ctl_fct = 0   # 0: linear control function, 1: optimal control function

    nb_pts_gap = 20     # number of points for the gap computation
    nb_pts_time = 30    # number of points for resolution of the time dependant Schrodinger's equation
    epsilon = 0.1       # precision level for the control function (valid for both, linear and optimal scheduling)
    gamma = 1           # strength of the transverse field, irrelevant for us 

    path = "./annealing_data/"
    append_to_existing_data = True  # if True, the newly generated data will be appended to the already existing files

    # -------------- Program -----------------
    thres_min = float(thres_min)                # to ensure the filenames are always with e.g. in case of thres_min=0: 0.0 and not sometimes 0 but sometimes 0.0
    thres_max = float(thres_max)

    if append_to_existing_data == False:
        warn_user(path)

    # number of runs
    N_neglection_thres = int((thres_max-thres_min)/thres_step)+1
    N_annealing_runs = N_neglection_thres*N_per_thres
    print(f"{N_neglection_thres} neglection thresolds and {N_annealing_runs} annealing runs")

    # activity pattern
    alpha = np.zeros(N, dtype=int)
    alpha[0] = 1
    alpha[1] = 1


    # runtime estimation
    t0 = time.time()
    for i in range(10):
        J, b, *_ = get_ising_parameters(N, M, alpha, K, xi, False)
        J_n, _ = modify_coupling_matrix(J, 1, 0.1, False)
        _, _, _, _, _, _, _, _, Hscheduled = solve_annealing(J, b, gamma, epsilon, which_ctl_fct, nb_pts_gap, nb_pts_time, False)
        _, _, gs_array = get_groundstate(Hscheduled, 1)
    t1 = time.time()

    print(f"estimated time for generation of file for one neglection thresold: {(t1-t0)/10*N_per_thres/60:.2f} min")
    estimated_runtime = (t1-t0)/10*N_annealing_runs/60
    print(f"estimated runtime of program: {estimated_runtime:.2f} min = {estimated_runtime/60:.2f} h")


    # actual data generation
    for i in range(N_neglection_thres):
        ti0 = time.time()

        # calculate the new neglection thresold
        neglection_thres = thres_min + i*thres_step
        neglection_thres = np.round(neglection_thres, 6) # without rounding, there were sometimes thresolds like 0.21000000000000002 which got then written as names on the datafiles

        # generate data
        J_data, J_n_data, alpha_data, gs_array_data, where_n_data = generate_data_for_one_thres(neglection_thres, N_per_thres, N, M, alpha, K, xi, gamma, epsilon, which_ctl_fct, nb_pts_gap, nb_pts_time)

        # check if a file of this configuration already exists - if yes, append the data instead of overwriting (if selected)
        fname = path + f"neglection_thres_{neglection_thres}_N_{N}_M_{M}_K_{K}_xi_{xi}.npy"
        if append_to_existing_data:
            try:
                loaded_dict = np.load(fname, allow_pickle=True).item()
                append_str = "(appended to existing file)"

                J_data          = np.concatenate((loaded_dict["J"],    J_data), axis=0)
                J_n_data        = np.concatenate((loaded_dict["J"],  J_n_data), axis=0)
                gs_array_data   = np.concatenate((loaded_dict["gs_array"],  gs_array_data), axis=0)
                alpha_data      = np.concatenate((loaded_dict["alpha"],     alpha_data))
                where_n_data    = [*loaded_dict["where_n"], *where_n_data]


            except FileNotFoundError:
                append_str = "(file not found -> new file created)"

        else:
            append_str = "(append_to_existing_data = False -> new file generated)"

        # build the dictionary
        data_dict = {
            "J":                J_data,
            "J_n":              J_n_data,
            "where_n":          where_n_data,
            "gs_array":         gs_array_data,
            "neglection_thres": neglection_thres,
            "alpha":            alpha_data,
            "size":             len(where_n_data),

            "N":                N,
            "M":                M,
            "K":                K,
            "xi":               xi,
            "epsilon":          epsilon,
            "gamma":            gamma,
            "nb_pts_gap":       nb_pts_gap,
            "nb_pts_time":      nb_pts_time,
            "which_ctl_fct":    which_ctl_fct,
        }

        # save the dictionary
        fname = f"./annealing_data/neglection_thres_{neglection_thres}_N_{N}_M_{M}_K_{K}_xi_{xi}.npy"
        np.save(fname, data_dict)
        ti1 = time.time()

        print(f"{i} of {N_neglection_thres} (runtime: {(ti1-ti0)/60:.2f} min): Data saved as {fname}    {append_str}")

    t2 = time.time()
    actual_runtime = (t2-t0)/60
    print(f"actual runtime of the program: {actual_runtime:.2f} min       (estimated runtime of program was: {estimated_runtime:.2f} min) (actual/estimated = {actual_runtime/estimated_runtime:.2f})")



def main():
    generate_data_and_save()

    fname = f"./annealing_data/neglection_thres_{0.0}_N_{5}_M_{4}_K_{100}_xi_{0}.npy"
    check_data(fname)
    fname = f"./annealing_data/neglection_thres_{0.4}_N_{5}_M_{4}_K_{100}_xi_{0}.npy"
    check_data(fname)


if __name__ == "__main__":
    main()
