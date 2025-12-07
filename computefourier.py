import math
import cmath
import numpy as np
from scipy import fftpack
from scipy.fftpack import fft, fftshift, ifft
from utils import *
from filters import *

DEBUG = False  # set True if you want extra prints


def Comb_Filt(origx, n, num, W_Comb):
    """
    Randomly samples W_Comb elements from input array, performs an FFT,
    and returns indices of the 'num' largest frequencies.
    """
    n = int(n)
    W_Comb = int(W_Comb)

    if n % W_Comb:
        raise Exception("W_Comb is not divisible by N, which algorithm expects")

    x_sampt = np.zeros(W_Comb, dtype=complex)

    # use integer division
    sigma = n // W_Comb
    offset = int(np.random.rand(1)[0] * sigma)

    for i in range(W_Comb):
        x_sampt[i] = origx[offset + i * sigma]

    x_sampt = fftpack.fft(x_sampt, W_Comb)
    samples = np.zeros(W_Comb, dtype=float)

    for i in range(W_Comb):
        samples[i] = cabs2(x_sampt[i])

    Comb_Approved = find_largest_indices(num, samples)

    if DEBUG:
        print(f"[Comb_Filt] n={n}, W_Comb={W_Comb}, num={num}, "
              f"top={len(Comb_Approved)}")

    return Comb_Approved


def inner_loop_locate(origx, n, current_filter, num, B, a, ai, b):
    """
    Apply time-domain filter, fold into B buckets, FFT, and pick top 'num'.
    """
    B = int(B)
    n = int(n)
    a = int(a)
    ai = int(ai)
    b = int(b)

    if n % B:
        print("Warning: n is not divisible by B")

    x_sampt = np.zeros(B, dtype=complex)
    index = b  # start index (int)

    for i in range(current_filter.sig_t.shape[0]):
        x_sampt[i % B] += origx[int(index)] * current_filter.sig_t[i]
        index = (index + ai) % n
        index = int(index)

    x_samp_i = fftpack.fft(x_sampt, B)
    samples = np.power(np.abs(x_samp_i), 2)

    J = find_largest_indices(num, samples)

    if DEBUG:
        print(f"[inner_loop_locate] B={B}, num={num}, len(J)={len(J)}")

    return x_samp_i, J


def inner_loop_filter_regular(J, n, num, B, a, ai, b,
                              loop_threshold, score, hits, hits_found):
    """
    Voting scheme without comb filter.
    """
    n = int(n)
    B = int(B)

    for i in range(num):

        low = (int(math.ceil((J[i] - 0.5) * n / B)) + n) % n
        high = (int(math.ceil((J[i] + 0.5) * n / B)) + n) % n
        loc = (low * a) % n

        j = low
        while j != high:

            score[loc] += 1
            if score[loc] == loop_threshold:
                hits[hits_found] = loc
                hits_found += 1

            loc = (loc + a) % n
            j = (j + 1) % n

    if DEBUG:
        print(f"[inner_loop_filter_regular] hits_found={hits_found}")

    return hits_found


def inner_loop_filter_Comb(J, n, num, B, a, ai, b,
                           loop_threshold, score, hits, hits_found,
                           Comb_Approved, num_Comb, W_Comb):
    """
    Voting scheme with comb-filter candidate pruning.
    """
    n = int(n)
    B = int(B)
    W_Comb = int(W_Comb)

    permuted_approved = np.zeros((num_Comb, 2), dtype=int)

    for m in range(num_Comb):
        prev = (Comb_Approved[m] * ai) % W_Comb
        permuted_approved[m][0] = prev
        permuted_approved[m][1] = (prev * a) % n

    permuted_approved = permuted_approved[np.argsort(permuted_approved[:, 0])]

    for i in range(num):

        low = (int(math.ceil((J[i] - 0.5) * n / B)) + n) % n
        high = (int(math.ceil((J[i] + 0.5) * n / B)) + n) % n

        index_arr = np.where(permuted_approved[:, 0] > (low % W_Comb))[0]
        if len(index_arr) == 0:
            index = num_Comb
        else:
            index = int(index_arr[0])

        location = low - (low % W_Comb)
        locinv = (location * a) % n
        j = index

        while True:

            if j == num_Comb:
                j -= num_Comb
                location = (location + W_Comb) % n
                locinv = (location * a) % n

            approved_loc = location + permuted_approved[j][0]

            # outside interval -> stop
            if ((low < high and (approved_loc >= high or approved_loc < low)) or
                (low > high and (approved_loc >= high and approved_loc < low))):
                break

            loc = (locinv + permuted_approved[j][1]) % n
            score[loc] += 1

            if score[loc] == loop_threshold:
                hits[hits_found] = loc
                hits_found += 1

            j += 1

    if DEBUG:
        print(f"[inner_loop_filter_Comb] hits_found={hits_found}")

    return hits_found


def estimate_values(hits, hits_found, x_samp, loops, n,
                    permute, B, B2, filter_loc, filter_est, location_loops):
    """
    Given candidate indices 'hits', estimate complex values by combining
    measurements over 'loops' using a median-like statistic.
    """
    n = int(n)
    B = int(B)
    B2 = int(B2)
    loops = int(loops)
    location_loops = int(location_loops)

    ans = {}
    values = np.zeros((2, loops), dtype=float)

    for i in range(hits_found):

        position = 0

        for j in range(loops):

            if j < location_loops:
                cur_B = B
                current_filter = filter_loc
            else:
                cur_B = B2
                current_filter = filter_est

            cur_B = int(cur_B)
            segment = n // cur_B  # bin size

            permuted_index = (permute[j] * hits[i]) % n
            hashed_to = permuted_index // segment
            dist = permuted_index % segment

            if dist > (segment // 2):
                hashed_to = (hashed_to + 1) % cur_B
                dist -= segment

            dist = (n - dist) % n

            filter_value = current_filter.sig_f[dist]
            values[0][position] = (x_samp[j][hashed_to] / filter_value).real
            values[1][position] = (x_samp[j][hashed_to] / filter_value).imag
            position += 1

        # median index (integer)
        location = (loops - 1) // 2

        for a_idx in range(2):
            values[a_idx] = nth_element(values[a_idx], location)

        realv = values[0][location]
        imagv = values[1][location]

        ans[hits[i]] = complex(realv, imagv)

    if DEBUG:
        print(f"[estimate_values] estimated {len(ans)} coefficients")

    return ans


def outer_loop(origx, n, filter_loc, filter_est,
               B2, num, B, W_Comb,
               Comb_loops, loop_threshold,
               location_loops, loops, ALG_TYPE):
    """
    Top-level sparse FFT loop.

    outer_loop(x, n, filter_loc, filter_est, B_est, B_thresh,
               B_loc, W_Comb, comb_loops, threshold_loops,
               loc_loops, loc_loops + est_loops)
    """
    n = int(n)
    B = int(B)
    B2 = int(B2)
    W_Comb = int(W_Comb)
    loops = int(loops)
    location_loops = int(location_loops)

    permute = np.zeros(loops, dtype=int)
    permuteb = np.zeros(loops, dtype=int)

    # list of 1D arrays (some length B, some length B2)
    x_samp = []

    hits_found = 0
    hits = np.zeros(n, dtype=int)
    score = np.zeros(n, dtype=int)
    num_Comb = num
    Comb_Approved = []

    if ALG_TYPE == 2:
        # WITH_COMB
        for i in range(Comb_loops):
            num_largest = Comb_Filt(origx, n, num, W_Comb)
            Comb_Approved.append(num_largest)

        Comb_Approved = np.array(Comb_Approved)
        Comb_Approved = np.reshape(Comb_Approved, (-1,))
        Comb_Approved = np.unique(Comb_Approved)
        num_Comb = Comb_Approved.shape[0]

        blocks = n // W_Comb
        hits_found = num_Comb * blocks

        for j in range(blocks):
            for i in range(num_Comb):
                hits[j * num_Comb + i] = j * W_Comb + Comb_Approved[i]

        if DEBUG:
            print(f"[outer_loop] ALG_TYPE=2, num_Comb={num_Comb}, "
                  f"blocks={blocks}, initial hits_found={hits_found}")

    # inner loops
    for i in range(loops):

        a = 0
        b = 0

        while gcd(a, n) != 1:
            a = np.random.randint(0, n, 1)[0]

        ai = mod_inverse(a, n)

        permute[i] = ai
        permuteb[i] = b

        perform_location = int(i < location_loops)

        if perform_location:
            current_filter = filter_loc
            current_B = B
        else:
            current_filter = filter_est
            current_B = B2

        x_samp_i, J = inner_loop_locate(
            origx, n, current_filter, num, current_B, a, ai, b
        )
        x_samp.append(x_samp_i)

        if perform_location:
            if ALG_TYPE == 1:
                hits_found = inner_loop_filter_regular(
                    J, n, num, current_B, a, ai, b,
                    loop_threshold, score, hits, hits_found
                )
            else:
                hits_found = inner_loop_filter_Comb(
                    J, n, num, current_B, a, ai, b,
                    loop_threshold, score, hits, hits_found,
                    Comb_Approved, num_Comb, W_Comb
                )

        if DEBUG:
            print(f"[outer_loop] loop {i+1}/{loops}, hits_found={hits_found}")

    # IMPORTANT: keep x_samp as list (B and B2 differ), do NOT np.array it
    ans = estimate_values(
        hits, hits_found, x_samp, loops, n,
        permute, B, B2, filter_loc, filter_est, location_loops
    )

    if DEBUG:
        print(f"[outer_loop] Finished with {len(ans)} non-zero coefficients")

    return ans
