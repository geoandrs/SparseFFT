import argparse
import math
import time
import timeit

import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

from filters import *
from utils import *
from computefourier import *

# Default parameter values that can be changed via command line
GRAPH_TYPE = 1        # 1: vary N, 2: vary K, 3: vary SNR
REPETITIONS = 1
VERBOSE = True
ALG_TYPE = 1          # 1: original, 2: with comb-filter

DEBUG = False
VISUALIZE = True
GENERATE_PLOTS = True


def run_experiment(x, x_f, large_freq, k, n,
                   lobefrac_loc, tolerance_loc, b_loc, B_loc, B_thresh,
                   loc_loops, threshold_loops,
                   lobefrac_est, tolerance_est, b_est, B_est,
                   est_loops, W_Comb, comb_loops, snr):
    """
    Runs the sparse FFT experiment once and returns (runtime, error_per_entry).
    """
    if DEBUG:
        print(f"[run_experiment] n={n}, k={k}")
        print(f"  lobefrac_loc={lobefrac_loc}, tolerance_loc={tolerance_loc}")
        print(f"  lobefrac_est={lobefrac_est}, tolerance_est={tolerance_est}")
        print(f"  b_loc={b_loc}, B_loc={B_loc}, B_thresh={B_thresh}")
        print(f"  b_est={b_est}, B_est={B_est}")
        print(f"  loc_loops={loc_loops}, est_loops={est_loops}, "
              f"threshold_loops={threshold_loops}, comb_loops={comb_loops}")
        print(f"  W_Comb={W_Comb}")

    # build filters
    filter_t = chebyshev_window(lobefrac_loc, tolerance_loc)
    filter_loc = make_multiple(filter_t, n, b_loc)

    filter_t = chebyshev_window(lobefrac_est, tolerance_est)
    filter_est = make_multiple(filter_t, n, b_est)

    # main sFFT outer loop
    start_time = time.perf_counter()
    ans = outer_loop(
        x, n, filter_loc, filter_est,
        B_est, B_thresh, B_loc, W_Comb,
        comb_loops, threshold_loops,
        loc_loops, loc_loops + est_loops,
        ALG_TYPE,
    )
    end_time = time.perf_counter()
    t = end_time - start_time

    num_candidates = len(ans)
    candidates = np.zeros((num_candidates, 2))
    x_f_Large = np.zeros(n, dtype=complex)
    ans_Large = np.zeros(n, dtype=complex)
    counter = 0
    ERROR = 0.0

    for key in sorted(ans.keys()):
        value = ans[key]
        candidates[counter][1] = int(key)
        candidates[counter][0] = abs(value)
        counter += 1

    for i in range(k):
        x_f_Large[large_freq[i]] = x_f[large_freq[i]]

    tmp = np.argpartition(candidates[:, 0], num_candidates - k)
    candidates = candidates[tmp]

    for l in range(k):
        key = int(candidates[num_candidates - k + l][1])
        ans_Large[key] = ans[key]

    for i in range(n):
        ERROR += abs(ans_Large[i] - x_f_Large[i])

    print(
        "Experiment N= %d, k=%d, ERROR = %3.9f, ERROR per entry = %3.9f"
        % (n, k, ERROR, ERROR / k)
    )

    err_vec = np.sqrt(
        np.power(
            np.subtract(np.abs(ans_Large), np.abs(x_f_Large)),
            2,
        )
    )

    if GENERATE_PLOTS:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))

        plt.subplot(2, 2, 1)
        plt.title("Signal Time Domain")
        plt.plot(np.abs(x))

        plt.subplot(2, 2, 2)
        plt.title("DFT Signal")
        plt.plot(np.abs(x_f), linewidth=0.3)

        plt.subplot(2, 2, 3)
        plt.title("Sparse FFT of The Signal")
        plt.plot(np.abs(ans_Large), linewidth=0.3)

        plt.subplot(2, 2, 4)
        plt.title("Error Vector")
        plt.plot(err_vec)

        if GRAPH_TYPE != 3:
            plt.suptitle(f"Sparse FFT: N={n}, k={k}")
        else:
            plt.suptitle(f"Sparse FFT: N={n}, k={k}, SNR={snr} dB")

        # We only show here if you have a GUI backend; saving is done in main.
        plt.tight_layout()
        plt.show(block=False)
        plt.close(fig)

    ERROR = ERROR / k
    return t, ERROR


def main():
    global GRAPH_TYPE, REPETITIONS, ALG_TYPE, VERBOSE

    print("[INFO] Starting main() of generate_graphs.py")

    parser = argparse.ArgumentParser(description="Sparse FFT experiments")
    parser.add_argument(
        "-g", "--graph_type",
        help="Type of graph (1: vs N, 2: vs K, 3: vs SNR)",
        type=int,
        required=False,
    )
    parser.add_argument(
        "-r", "--repetitions",
        help="Number of repetitions",
        type=int,
        required=False,
    )
    parser.add_argument(
        "-v", "--verbose",
        help="Verbose mode",
        type=bool,
        required=False,
    )
    parser.add_argument(
        "-a", "--alg_type",
        help="sFFT 1.0 (1) or sFFT 2.0 (2)",
        type=int,
        required=False,
    )

    args = vars(parser.parse_args())

    if args["graph_type"] is not None:
        g = args["graph_type"]
        if 1 <= g <= 3:
            GRAPH_TYPE = g
            print(f"[INFO] Program arg graph_type: {GRAPH_TYPE}")
        else:
            print("graph_type: Invalid option")

    if args["repetitions"] is not None:
        r = args["repetitions"]
        if r >= 1:
            REPETITIONS = r
            print(f"[INFO] Program arg repetitions: {REPETITIONS}")
        else:
            print("repetitions: Invalid option")

    if args["verbose"] is not None:
        VERBOSE = args["verbose"]
        print(f"[INFO] Verbose: {VERBOSE}")

    if args["alg_type"] is not None:
        a = args["alg_type"]
        if a in (1, 2):
            ALG_TYPE = a
            print(f"[INFO] ALG_TYPE: {ALG_TYPE}")
        else:
            print("alg_type: Invalid option")

    # -------- experiment parameter grids --------
    N_vec = np.array(
        [
            8192,
            16384,
            32768,
            65536,
            131072,
            262144,
            524288,
            1048576,
        ],
        dtype=int,
    )
    K_vec = np.array([50, 100, 200, 500, 1000, 2000], dtype=int)
    SNR_vec = np.array(
        [-20, -10, -7, -3, 0, 3, 7, 10, 20, 30, 40, 50, 60, 120],
        dtype=float,
    )

    if GRAPH_TYPE == 1:
        length = len(N_vec)
    elif GRAPH_TYPE == 2:
        length = len(K_vec)
    else:
        length = len(SNR_vec)

    print(f"[INFO] GRAPH_TYPE={GRAPH_TYPE}, length={length}, "
          f"REPETITIONS={REPETITIONS}, ALG_TYPE={ALG_TYPE}")

    sFFT_times = np.zeros(length)
    fft_times = np.zeros(length)
    sFFT_errors = np.zeros(length)

    snr = 100.0

    for i in range(length):
        print(f"[INFO] ===== Outer index i={i+1}/{length} =====")

        if GRAPH_TYPE == 1:
            n = int(N_vec[i])
            k = 50
            experiment_parameters = get_expermient_vs_N_parameters(n, ALG_TYPE)

        elif GRAPH_TYPE == 2:
            n = 1048576
            k = int(K_vec[i])
            experiment_parameters = get_expermient_vs_K_parameters(k, ALG_TYPE)

        else:
            n = 262144
            k = 50
            snr = float(SNR_vec[i])
            experiment_parameters = get_expermient_vs_N_parameters(n, ALG_TYPE)

        # unpack experiment parameters
        BCST_LOC = experiment_parameters.get("Bcst_loc", None)
        BCST_EST = experiment_parameters.get("Bcst_est", None)
        COMB_CST = experiment_parameters.get("Comb_cst", None)
        COMB_LOOPS = experiment_parameters.get("comb_loops", None)
        EST_LOOPS = experiment_parameters.get("est_loops", None)
        LOC_LOOPS = experiment_parameters.get("loc_loops", None)
        THRESHOLD_LOOPS = experiment_parameters.get("threshold_loops", None)
        TOLERANCE_LOC = experiment_parameters.get("tolerance_loc", None)
        TOLERANCE_EST = experiment_parameters.get("tolerance_est", None)

        if None in (
            BCST_LOC,
            BCST_EST,
            COMB_CST,
            COMB_LOOPS,
            EST_LOOPS,
            LOC_LOOPS,
            THRESHOLD_LOOPS,
            TOLERANCE_LOC,
            TOLERANCE_EST,
        ):
            raise ValueError("Experiment parameters missing some entries")

        BB_loc = math.floor(
            BCST_LOC * np.sqrt((n * k) / np.log2(n))
        )
        BB_est = math.floor(
            BCST_EST * np.sqrt((n * k) / np.log2(n))
        )

        LOBEFRAC_LOC = 0.5 / BB_loc
        LOBEFRAC_EST = 0.5 / BB_est

        b_loc = int(1.2 * 1.1 * (n / BB_loc))
        b_est = int(1.4 * 1.1 * (n / BB_est))

        B_loc = floor_to_pow2(BB_loc)
        B_thresh = 2 * k
        B_est = floor_to_pow2(BB_est)

        W_Comb = floor_to_pow2(COMB_CST * n / B_loc)

        if DEBUG:
            print(f"[main] n={n}, k={k}, B_loc={B_loc}, B_est={B_est}, "
                  f"W_Comb={W_Comb}, BB_loc={BB_loc}, BB_est={BB_est}")

        for j in range(REPETITIONS):
            print(f"[INFO]  Repetition {j+1}/{REPETITIONS}")

            x, x_f, large_freq = generate_random_signal(n, k)

            if GRAPH_TYPE == 3:
                std_noise = math.sqrt(k / (2.0 * math.pow(10, snr / 10)))
                snr_achieved = AWGN(x, n, std_noise)
                x_f = fftpack.fft(x, n) / n
                print(f"[INFO]   Target SNR: {snr} dB, achieved {snr_achieved:.6f} dB")

            wrapped = wrapper(fftpack.fft, x, n)
            fft_time = timeit.timeit(wrapped, number=1)
            fft_times[i] += fft_time

            print("-------------------------------------------------------------------------")
            sfft_time, error = run_experiment(
                x,
                x_f,
                large_freq,
                k,
                n,
                LOBEFRAC_LOC,
                TOLERANCE_LOC,
                b_loc,
                B_loc,
                B_thresh,
                LOC_LOOPS,
                THRESHOLD_LOOPS,
                LOBEFRAC_EST,
                TOLERANCE_EST,
                b_est,
                B_est,
                EST_LOOPS,
                W_Comb,
                COMB_LOOPS,
                snr,
            )

            sFFT_times[i] += sfft_time
            sFFT_errors[i] += error

            print(f"Sfft time: {sfft_time:0.6f} sec")
            print(f"FFT  time: {fft_time:0.6f} sec")
            print("-------------------------------------------------------------------------")

    # average over repetitions
    sFFT_errors = sFFT_errors / REPETITIONS
    fft_times = fft_times / REPETITIONS
    sFFT_times = sFFT_times / REPETITIONS

    # ---- plotting & saving ----
    print("[INFO] Finished all experiments, creating plots and saving results...")

    if GRAPH_TYPE == 1:
        plt.figure()
        plt.plot(N_vec, sFFT_errors)
        plt.ylabel("Error")
        plt.xlabel("N")
        plt.yscale("log")
        plt.title("Error vs N, k=50")
        plt.savefig("error_vs_N_k50.png", dpi=200, bbox_inches="tight")
        print("[INFO] Saved plot: error_vs_N_k50.png")
        plt.close()

        plt.figure()
        plt.plot(N_vec, sFFT_times, label="Sparse FFT runtime")
        plt.plot(N_vec, fft_times, label="FFT runtimes")
        plt.title("Sparse FFT vs FFT Run Times, k=50")
        plt.ylabel("Runtime (sec)")
        plt.xlabel("N")
        plt.legend()
        plt.savefig("runtime_vs_N_k50.png", dpi=200, bbox_inches="tight")
        print("[INFO] Saved plot: runtime_vs_N_k50.png")
        plt.close()

    elif GRAPH_TYPE == 2:
        plt.figure()
        plt.plot(K_vec, sFFT_errors)
        plt.yscale("log")
        plt.ylabel("Error")
        plt.xlabel("K")
        plt.title("Error vs K, N = 1048576")
        plt.savefig("error_vs_K_N1048576.png", dpi=200, bbox_inches="tight")
        print("[INFO] Saved plot: error_vs_K_N1048576.png")
        plt.close()

        plt.figure()
        plt.plot(K_vec, sFFT_times, label="Sparse FFT runtime")
        plt.plot(K_vec, fft_times, label="FFT runtimes")
        plt.title("Sparse FFT vs FFT Run Times, N=1048576")
        plt.ylabel("Runtime (sec)")
        plt.xlabel("K")
        plt.legend()
        plt.savefig("runtime_vs_K_N1048576.png", dpi=200, bbox_inches="tight")
        print("[INFO] Saved plot: runtime_vs_K_N1048576.png")
        plt.close()

    else:
        plt.figure()
        plt.plot(SNR_vec, sFFT_errors)
        plt.yscale("log")
        plt.ylabel("Error")
        plt.xlabel("SNR (dB)")
        plt.title("Error vs SNR, N=262144, k=50")
        plt.savefig("error_vs_SNR_N262144_k50.png", dpi=200, bbox_inches="tight")
        print("[INFO] Saved plot: error_vs_SNR_N262144_k50.png")
        plt.close()

        plt.figure()
        plt.plot(SNR_vec, sFFT_times, label="Sparse FFT runtime")
        plt.plot(SNR_vec, fft_times, label="FFT runtimes")
        plt.title("Sparse FFT vs FFT Run Times, N=262144, k=50")
        plt.ylabel("Runtime (sec)")
        plt.xlabel("SNR (dB)")
        plt.legend()
        plt.savefig("runtime_vs_SNR_N262144_k50.png", dpi=200, bbox_inches="tight")
        print("[INFO] Saved plot: runtime_vs_SNR_N262144_k50.png")
        plt.close()

    # Save raw numeric results
    out_file = f"results_graph_type_{GRAPH_TYPE}.npz"
    np.savez(
        out_file,
        GRAPH_TYPE=GRAPH_TYPE,
        N_vec=N_vec,
        K_vec=K_vec,
        SNR_vec=SNR_vec,
        sFFT_errors=sFFT_errors,
        sFFT_times=sFFT_times,
        fft_times=fft_times,
    )
    import os
    print(f"[INFO] Saved numeric results to: {os.path.abspath(out_file)}")
    print("[INFO] All done.")


if __name__ == "__main__":
    main()
