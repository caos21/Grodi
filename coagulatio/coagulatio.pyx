#cython: np_pythran=False
#cython: language_level=3
#
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
cimport cython
cimport numpy as np
import numpy as np
cimport openmp
from cython.parallel cimport prange
from cython.parallel cimport parallel

@cython.boundscheck(False)
@cython.wraparound(False)
def deathlist_toarrays(list death_list,
                       short[:] larr, short[:] qarr,
                       short[:] marr, short[:] parr,
                       double[:] death_array):
    cdef unsigned int N = len(death_list)
    cdef unsigned int i
    for i in range(N):
      larr[i] = death_list[i][1]
      qarr[i] = death_list[i][2]
      marr[i] = death_list[i][3]
      parr[i] = death_list[i][4]
      death_array[i] = death_list[i][5]

@cython.boundscheck(False)
@cython.wraparound(False)
def birthlist_toarrays(list birth_list,
                       short[:] larr, short[:] qarr,
                       short[:] marr, short[:] parr,
                       short[:] narr, short[:] rarr,
                       double[:] birth_array):
    cdef unsigned int N = len(birth_list)
    cdef unsigned int i
    for i in range(N):
      larr[i] = birth_list[i][1]
      qarr[i] = birth_list[i][2]
      marr[i] = birth_list[i][3]
      parr[i] = birth_list[i][4]
      narr[i] = birth_list[i][5]
      rarr[i] = birth_list[i][6]
      birth_array[i] = birth_list[i][7]
    
@cython.boundscheck(False)
@cython.wraparound(False)
def serial_death(short[:] l, short[:] q,
                 short[:] m, short[:] p,
                 double[:] eta,
                 double[:, :] dens,
                 double[:, :] res):
  cdef unsigned int N = eta.shape[0]
  cdef unsigned int i
  for i in range(N):
    res[l[i], q[i]] += eta[i]*dens[m[i], p[i]]
    
@cython.boundscheck(False)
@cython.wraparound(False)
def parallel_death(short[:] l, short[:] q,
                   short[:] m, short[:] p,
                   double[:] eta,
                   double[:, :] dens,
                   double[:, :] res):
  cdef unsigned int N = eta.shape[0]
  cdef unsigned int i
  with nogil:
    for i in prange(N, schedule='static'):
      res[l[i], q[i]] += eta[i]*dens[m[i], p[i]]
    
@cython.boundscheck(False)
@cython.wraparound(False)
def serial_birth(short[:] l, short[:] q,
                 short[:] m, short[:] p,
                 short[:] n, short[:] r,
                 double[:] eta,
                 double[:, :] dens,
                 double[:, :] res):
  cdef unsigned int N = eta.shape[0]
  cdef unsigned int i
  for i in range(N):
    res[l[i], q[i]] += eta[i]*dens[m[i], p[i]]*dens[n[i], r[i]]
    
@cython.boundscheck(False)
@cython.wraparound(False)
def parallel_birth(short[:] l, short[:] q,
                   short[:] m, short[:] p,
                   short[:] n, short[:] r,
                   double[:] eta,
                   double[:, :] dens,
                   double[:, :] res):
  cdef unsigned int N = eta.shape[0]
  cdef unsigned int i
  with nogil:
    for i in prange(N, schedule='static'):
      res[l[i], q[i]] += eta[i]*dens[m[i], p[i]]*dens[n[i], r[i]]
