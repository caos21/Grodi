# charging.pyx
#cython: language_level=3
#   Copyright 2019 Benjamin Santos
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# -*- coding: utf-8 -*-
from __future__ import print_function

from libc.stdio cimport printf
from libc.stdlib cimport malloc, free

cimport cython
cimport numpy as np
import numpy as np
cimport openmp
from cython.parallel cimport prange
from cython.parallel cimport parallel

cimport ccharging

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double* assign1D(double [:] nparray1D):
    rows = np.size(nparray1D)
    cdef double * carray1D = <double *>malloc(rows * sizeof(double))
    if carray1D == NULL:
        printf("[ee] Error allocating memory (malloc)\n")
        return NULL

    cdef int i
    
    for i in range(rows):
        carray1D[i]= nparray1D[i]
    return carray1D

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double** assign2D(double [:,:] nparray2D):
    rows, cols = np.shape(nparray2D)
    cdef double ** carray2D = <double **>malloc(rows * sizeof(double*))
    if carray2D == NULL:
        print("[ee] Error allocating memory (malloc)\n")
        return NULL;\

    cdef int i
    for i in range(rows):
        carray2D[i] = <double *>malloc(cols * sizeof(double*))
        if carray2D == NULL:
            print("[ee] Error allocating memory (malloc)\n");
            return NULL
    
    for i in range(rows):
        for j in range(cols):
            carray2D[i][j] = nparray2D[i][j]
    return carray2D

@cython.boundscheck(False)
@cython.wraparound(False)
cdef reassign2D(double [:,:] nparray2D,
                double ** carray2D):
    rows, cols = np.shape(nparray2D)

    for i in range(rows):
        for j in range(cols):
            nparray2D[i][j] = carray2D[i][j]

#@cython.boundscheck(False)
#@cython.wraparound(False)
def charging_totalstep(time, delta_t, grid_data, pc, gd, charging,
                       plasmasystem):
    """ Performs a charging step
        inputs time, delta_t, pc, grid_data, gd, charging, plasmasystem
    """
    cdef double tin = time
    cdef double dt = delta_t

    cdef double* pdensity = assign1D(pc.next_plasmadensity)
    cdef double edensity = pdensity[0]
    cdef double energy = pdensity[6]/edensity
    cdef double idensity = pdensity[1]

    charging.compute_freqs(energy, edensity, idensity)

    cdef short nvols = grid_data.nvols
    cdef short nchrgs = grid_data.nchrgs
    
    cdef double with_tunnel = plasmasystem.with_tunnel

    # define collision data
    cdef ccharging.collision_data mcd
    mcd.l = 0
    mcd.nchrgs = nchrgs
    mcd.with_tunnel = with_tunnel
    mcd.efreq = assign2D(charging.efreq)
    mcd.ifreq = assign2D(charging.ifreq)
    mcd.tfreq = assign2D(charging.tfreq)
       
    # define plasma density data
    cdef ccharging.plasma_data mpd
    mpd.nvols = nvols
    mpd.nchrgs = nchrgs    
    mpd.pdensity = pdensity
    mpd.pdens = assign2D(gd.past_density)
    mpd.ndens = assign2D(gd.next_density)
    mpd.qrate2d = assign2D(gd.qrate2d)    
   
    with nogil:
        ret = ccharging.charging_step(tin, dt, &mpd, &mcd)

    reassign2D(gd.next_density, mpd.ndens)
    reassign2D(gd.qrate2d, mpd.qrate2d)
            
    gd.next_density = np.nan_to_num(gd.next_density)
    gd.next_density[gd.next_density < 0.0] = 0.0

    gd.past_density = np.copy(gd.next_density)
    
    nano_qdens = np.sum(gd.next_density*grid_data.qpivots)
    pc.nano_qdens = nano_qdens
    
    nano_qdens_rate = np.sum(gd.qrate2d*grid_data.qpivots)
    pc.nano_qdens_rate = nano_qdens_rate    
    
    pc.next_plasmadensity[3] = pc.next_plasmadensity[0]-pc.nano_qdens-pc.next_plasmadensity[1]
    
    free(pdensity)

    free(mcd.efreq)
    free(mcd.ifreq)
    free(mcd.tfreq)
    
    free(mpd.pdens)
    free(mpd.ndens)
    free(mpd.qrate2d)
    return 0
