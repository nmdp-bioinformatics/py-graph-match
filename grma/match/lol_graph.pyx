#cython: language_level=3

import cython
from cpython cimport tuple
import numpy as np
cimport numpy as np

ctypedef np.int32_t INT
ctypedef np.uint32_t UINT
ctypedef np.uint16_t UINT16
ctypedef np.float32_t FLOAT

cdef class LolGraph:
    cdef:
        UINT[:] _index_list
        UINT[:] _neighbors_list
        FLOAT[:] _weights_list
        INT[:] _map_number_to_num_node
        UINT[:, :] _map_number_to_arr_node  # Changed from UINT16[:, :]
        UINT _arrays_start
        bint directed
        bint weighted

    def __init__(self, UINT[:] index_list,
                 UINT[:] neighbors_list,
                 FLOAT[:] weights_list,
                 INT[:] map_number_to_num_node,
                 UINT[:, :] map_number_to_arr_node,  # Changed from UINT16[:, :]
                 UINT arrays_start,
                 bint directed, bint weighted):
        self._index_list = index_list
        self._neighbors_list = neighbors_list
        self._weights_list = weights_list
        self._map_number_to_num_node = map_number_to_num_node
        self._map_number_to_arr_node = map_number_to_arr_node
        self._arrays_start = arrays_start
        self.directed = directed
        self.weighted = weighted

    @property
    def array_start(self):
        return self._arrays_start

    cpdef bint is_directed(self):
        return self.directed

    cpdef bint is_weighted(self):
        return self.weighted

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef UINT[:] arr_node_value_from_id(self, UINT node_id):  # Changed from UINT16[:]
        return self._map_number_to_arr_node[node_id - self._arrays_start]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef INT num_node_value_from_id(self, UINT node_id):
        return self._map_number_to_num_node[node_id]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef INT binary_search(self, np.ndarray[UINT, ndim=1] arr, UINT x):
        """
        Iterative Binary Search Function
        It returns index of x in given array arr if present,
        else returns -1
        """

        cdef UINT low, high, mid
        low = 0
        high = len(arr) - 1
        mid = 0

        while low <= high:
            mid = (high + low) // 2

            # Check if x is present at mid
            if arr[mid] < x:
                low = mid + 1

            # If x is greater, ignore left half
            elif arr[mid] > x:
                high = mid - 1

            # If x is smaller, ignore right half
            else:
                return mid

        # If we reach here, then the element was not present
        return -1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef FLOAT get_edge_data(self, UINT node1, UINT node2) except -1:
        """return the weight between two edges"""
        cdef UINT idx, idx_end, i
        cdef INT node2_index
        cdef np.ndarray[UINT, ndim=1] node1_neighbors
        idx = self._index_list[node1]
        idx_end = self._index_list[node1 + 1]

        node1_neighbors = np.zeros(idx_end - idx, dtype=np.uint32)
        for i in range(idx, idx_end):
            node1_neighbors[i - idx] = self._neighbors_list[i]

        node2_index = self.binary_search(node1_neighbors, node2)
        if self.is_weighted() and node2_index != -1:
            return self._weights_list[idx + node2_index]
        return -1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple neighbors_weighted(self, UINT node):
        """return the neighbors for weighted graph"""
        cdef UINT idx, idx_end, i
        cdef np.ndarray[UINT, ndim=1] neighbors_list_id
        cdef np.ndarray[FLOAT, ndim=1] weights_list

        idx = self._index_list[node]
        idx_end = self._index_list[node + 1]

        neighbors_list_id = np.zeros(idx_end - idx, dtype=np.uint32)
        for i in range(idx, idx_end):
            neighbors_list_id[i - idx] = self._neighbors_list[i]

        weights_list = np.zeros(idx_end - idx, dtype=np.float32)
        for i in range(idx, idx_end):
            weights_list[i - idx] = self._weights_list[i]

        return neighbors_list_id, weights_list

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[UINT, ndim=1] neighbors_unweighted(self, UINT node):
        """return the neighbors for unweighted graph"""
        cdef UINT idx, idx_end, i
        cdef np.ndarray[UINT, ndim=1] neighbors_list_id

        idx = self._index_list[node]
        idx_end = self._index_list[node + 1]

        neighbors_list_id = np.zeros(idx_end - idx, dtype=np.uint32)
        for i in range(idx, idx_end):
            neighbors_list_id[i - idx] = self._neighbors_list[i]

        return neighbors_list_id

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple neighbors_2nd(self, UINT node):
        """return the second degree neighbors of a node - neighbors of neighbors."""
        cdef UINT idx, idx_end, i, j, pointer, neighbor_1st, idx_1st_neigh, idx_end_1st_neigh, neighbor_id
        cdef np.ndarray[UINT, ndim=1] neighbors_list_id, neighbors_id
        cdef UINT[:] arr  # Changed from UINT16[:]
        cdef np.ndarray[UINT, ndim=2] neighbors_value  # Changed from UINT16
        cdef UINT num_of_neighbors_2nd
        cdef UINT loci_len

        idx = self._index_list[node]
        idx_end = self._index_list[node + 1]

        neighbors_list_id = np.zeros(idx_end - idx, dtype=np.uint32)
        for i in range(idx, idx_end):
            neighbors_list_id[i - idx] = self._neighbors_list[i]
        num_of_neighbors_2nd = <UINT>self._weights_list[idx]

        neighbors_id = np.zeros(int(num_of_neighbors_2nd), dtype=np.uint32)
        pointer = 0

        for i in range(len(neighbors_list_id)):
            neighbor_1st = neighbors_list_id[i]
            idx_1st_neigh = self._index_list[neighbor_1st]
            idx_end_1st_neigh = self._index_list[neighbor_1st + 1]

            for j in range(idx_1st_neigh, idx_end_1st_neigh):
                neighbors_id[pointer] = self._neighbors_list[j]
                pointer += 1

        loci_len = <UINT> self._map_number_to_arr_node.shape[1]
        neighbors_value = np.zeros((num_of_neighbors_2nd, loci_len), dtype=np.uint32)  # Changed from uint16
        for i in range(len(neighbors_id)):
            neighbor_id = neighbors_id[i]
            arr = self.arr_node_value_from_id(neighbor_id)
            for j in range(loci_len):
                neighbors_value[i, j] = arr[j]

        return neighbors_id, neighbors_value
