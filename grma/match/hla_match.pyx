# grma/match/hla_match.pyx

import numpy as np
cimport numpy as np
cimport cython
from libc.stdint cimport uint32_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[np.uint32_t, ndim=1] _as_uint32_1d(object arr):
    cdef np.ndarray[np.uint32_t, ndim=1] out = np.asarray(arr, dtype=np.uint32)
    if not out.flags.c_contiguous:
        out = np.ascontiguousarray(out)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[np.uint32_t, ndim=2] _as_uint32_2d(object arr):
    cdef np.ndarray[np.uint32_t, ndim=2] out = np.asarray(arr, dtype=np.uint32)
    if not out.flags.c_contiguous:
        out = np.ascontiguousarray(out)
    return out

def batch_locuses_match_between_genos(pat, dons):
    """
    Returns: np.ndarray (M, 3) → [total_mismatched_loci, total_GvH_alleles, total_HvG_alleles]

    Here we treat:
      - 0 as "missing"
      - > 0 as real allele codes
    and we skip a locus only if BOTH patient and donor are completely missing there.
    """
    cdef np.ndarray[np.uint32_t, ndim=1] pat_np = _as_uint32_1d(pat)
    cdef np.ndarray[np.uint32_t, ndim=2] dons_np = _as_uint32_2d(dons)

    cdef Py_ssize_t m = dons_np.shape[0]
    cdef Py_ssize_t i, j
    cdef uint32_t a1, b1, a2, b2
    cdef set P, D
    cdef int gvh_i, hvg_i, mis_i
    cdef int total_mis = 0, total_gvh = 0, total_hvg = 0

    cdef np.ndarray[np.int32_t, ndim=2] result = np.empty((m, 3), dtype=np.int32)

    cdef uint32_t[::1] pat_view = pat_np
    cdef uint32_t[:, ::1] dons_view = dons_np

    for j in range(m):
        total_mis = total_gvh = total_hvg = 0

        for i in range(0, pat_np.shape[0], 2):
            a1 = pat_view[i]
            b1 = pat_view[i + 1]
            a2 = dons_view[j, i]
            b2 = dons_view[j, i + 1]

            # Skip only if BOTH patient and donor are completely missing at this locus
            if (a1 == 0 and b1 == 0) and (a2 == 0 and b2 == 0):
                continue

            # Build sets of real allele types
            P = set()
            D = set()
            if a1 > 0: P.add(a1)
            if b1 > 0: P.add(b1)
            if a2 > 0: D.add(a2)
            if b2 > 0: D.add(b2)

            gvh_i = len(P - D)
            hvg_i = len(D - P)
            mis_i = max(gvh_i, hvg_i)

            total_gvh += gvh_i
            total_hvg += hvg_i
            total_mis += mis_i

        result[j, 0] = total_mis   # total mismatched loci
        result[j, 1] = total_gvh   # total GvH allele types
        result[j, 2] = total_hvg   # total HvG allele types

    return result
