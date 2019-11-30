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
""" This module contains the classes functions and helpers to compute
nanoparticle growth.
"""

__author__ = "Benjamin Santos"
__copyright__ = "Copyright 2019"
__credits__ = ["Benjamin Santos"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Benjamin Santos"
__email__ = "caos21@gmail.com"
__status__ = "Beta"

import h5py
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import savgol_filter

import coagulatio as coag

class GridData(object):
    """ Open and store the grid data
    """
    def __init__(self, h5gridprefix, defpath):

        self.h5gridprefix = defpath + h5gridprefix

        # Read grid file
        gridfname = self.h5gridprefix + '.h5'
        self.gridfile = h5py.File(gridfname, 'r')

        ## group volumes
        gvols = self.gridfile.get("Volume_sections")

        ## volumes
        self.vols = np.array(gvols.get("Volumes"))

        ## pivots diameters in nanometres
        self.dpivots = np.array(gvols.get("Diameters"))*1E9

        ## group charges
        gchgs = self.gridfile.get("Charge_sections")

        self.qpivots = np.array(gchgs.get("Charges"))
        self.neutral_idx = gchgs.attrs['Max_negative']

        self.nvols = len(self.vols)
        self.nchrgs = len(self.qpivots)

        self.death_list = []
        self.death_larray = []
        self.death_qarray = []
        self.death_marray = []
        self.death_parray = []
        self.death_array = []

        self.birth_list = []
        self.birth_larray = []
        self.birth_qarray = []
        self.birth_marray = []
        self.birth_parray = []
        self.birth_narray = []
        self.birth_rarray = []
        self.birth_array = []

    def set_rates(self):
        """ Get the birth and death rates
        """
        ge_interaction = self.gridfile.get("Electrostatic_interaction")
        birth_factor_vector = ge_interaction['eta_factor_vector']
        death_factor_vector = ge_interaction['death_factor_vector']

        self.birth_list = np.asarray(birth_factor_vector).tolist()

        self.death_list = np.asarray(death_factor_vector).tolist()
        self.death_larray = np.zeros(len(death_factor_vector), dtype=np.int16)
        self.death_qarray = np.zeros(len(death_factor_vector), dtype=np.int16)
        self.death_marray = np.zeros(len(death_factor_vector), dtype=np.int16)
        self.death_parray = np.zeros(len(death_factor_vector), dtype=np.int16)
        self.death_array = np.zeros(len(death_factor_vector), dtype=np.float64)

        self.birth_larray = np.zeros(len(birth_factor_vector), dtype=np.int16)
        self.birth_qarray = np.zeros(len(birth_factor_vector), dtype=np.int16)
        self.birth_marray = np.zeros(len(birth_factor_vector), dtype=np.int16)
        self.birth_parray = np.zeros(len(birth_factor_vector), dtype=np.int16)
        self.birth_narray = np.zeros(len(birth_factor_vector), dtype=np.int16)
        self.birth_rarray = np.zeros(len(birth_factor_vector), dtype=np.int16)
        self.birth_array = np.zeros(len(birth_factor_vector), dtype=np.float64)

    def free_lists(self):
        """ Free birth and death lists
        """
        del self.birth_list
        del self.death_list

    def close(self):
        """ Close the grid file
        """
        self.gridfile.close()
        del self.gridfile

class SurfaceGrowth():
    """ Defines the surface growth
    """
    def __init__(self, grid_data, sgrowth_rate=12e-9):
        """
        """
        self.grid_data = grid_data
        self.sgrowth_rate = sgrowth_rate
        self.adim_srate = np.array(np.zeros_like(self.grid_data.dpivots))
        self.compute_adim_srate()

    def compute_adim_srate(self):
        """ Computes the surface growth rate
        """
        surface_area = 1e-18*np.pi*self.grid_data.dpivots**2
        surface_rate = self.sgrowth_rate * surface_area
        #adim_srate = np.zeros_like(self.grid_data.dpivots)
        self.adim_srate[0] = surface_rate[0] / (self.grid_data.vols[1]-self.grid_data.vols[0])
        self.adim_srate[1:-1] = (surface_rate[1:-1] /
                                 (self.grid_data.vols[2:]-self.grid_data.vols[1:-1]))
        self.adim_srate[-1] = (surface_rate[-2] /
                               (self.grid_data.vols[-1]-self.grid_data.vols[-2]))


    def compute_sgrowth(self, past_density, gsurface_growth):
        """ Compute surface growth
        """
        # newaxis to allow multiplication of column adim_srate for each charge
        # surface growth is independent of the charging
        gsurface_growth[1:][:] =\
           (self.adim_srate[0:-1, np.newaxis]*past_density[0:-1][:]
            -self.adim_srate[1:, np.newaxis]*past_density[1:][:])
        gsurface_growth[0][:] =\
            (-self.adim_srate[0, np.newaxis]*past_density[0][:])
        gsurface_growth[-1][:] =\
            (-self.adim_srate[-2, np.newaxis]*past_density[-2][:])

class Nucleation():
    """ Defines the nucleation rate
    """
    def __init__(self, grid_data, nrate=1e18):
        """
        """
        self.grid_data = grid_data
        self.nrate = nrate
        self.jnucleation = np.zeros([self.grid_data.nvols, self.grid_data.nchrgs])

    def compute_nucleation(self):
        """ Computes the nucleation, first section
        """
        self.jnucleation[0][self.grid_data.neutral_idx] = self.nrate

class GrowthData():
    """ Computes and stores the results
    """
    def __init__(self, grid_data, end_time=1.0, delta_t=5e-4, n0=1e10, skip=10,
                 wnu=1.0, wco=1.0, wsg=1.0):

        self.grid_data = grid_data

        time = np.arange(0.0, end_time, delta_t)
        self.delta_t = delta_t
        self.dtol = 1e-6
        self.time = time
        self.skip = skip
        self.stime = self.time[::skip]

        self.wnu, self.wco, self.wsg = wnu, wco, wsg

        self.density = np.zeros([self.grid_data.nvols, self.grid_data.nchrgs])
        self.density[0, self.grid_data.neutral_idx] = n0
        self.past_density = np.copy(self.density)
        self.next_density = np.zeros([self.grid_data.nvols, self.grid_data.nchrgs])

        self.qrate2d = np.zeros([self.grid_data.nvols, self.grid_data.nchrgs])

        self.gsurface_growth = np.zeros([self.grid_data.nvols, self.grid_data.nchrgs])
        self.jnucleation = np.zeros([self.grid_data.nvols, self.grid_data.nchrgs])

        self.birth2d = np.zeros([self.grid_data.nvols, self.grid_data.nchrgs])
        self.death2d = np.zeros([self.grid_data.nvols, self.grid_data.nchrgs])


        self.krate = np.zeros(len(self.stime))
        self.sgrate = np.zeros(len(self.stime))
        self.maxparticle = None
        self.peak_diameter = None
        self.total_number = np.zeros(len(self.stime))
        self.total_volume = np.zeros(len(self.stime))
        self.total_charge = np.zeros(len(self.stime))

        self.nano_history = [np.zeros([self.grid_data.nvols,
                                       self.grid_data.nchrgs])]*len(self.stime)
        self.krate_history = [np.zeros([self.grid_data.nvols,
                                        self.grid_data.nchrgs])]*len(self.stime)
        self.sgrate_history = [np.zeros([self.grid_data.nvols,
                                         self.grid_data.nchrgs])]*len(self.stime)

        self.ddens = None
        self.cdens = None

    def update_density(self):
        """ Update nanoparticle density
        """
        self.past_density[:] = self.next_density[:]

    def update_results(self, iteration):
        """ Update nanoparticle density
        """
        if iteration%self.skip == 0:
            self.store_results(iteration//self.skip)

    def prune_results(self):
        """ Prune results
        """
        if np.any(self.next_density < 0.0):
            #print("[WW] Negative nanoparticle density")
            self.next_density[self.next_density < 0.0] = self.past_density[self.next_density < 0.0]
        self.next_density[self.next_density < self.dtol] = 0.0

    def advance(self):
        """ advance
        """
        wnu, wco, wsg = self.wnu, self.wco, self.wsg
        self.next_density = ((self.past_density
                              + self.delta_t*(wco*self.birth2d
                                              +wnu*self.jnucleation
                                              +wsg*self.gsurface_growth))
                             /(1.0+wco*self.delta_t*self.death2d))

        self.prune_results()

    def store_results(self, i):
        """ Store results (funcion of time)
        """
        self.krate[i] = np.sum(np.sum(self.birth2d, axis=1)*self.grid_data.vols)
        self.sgrate[i] = np.sum(np.sum(np.abs(self.gsurface_growth), axis=1)*self.grid_data.vols)
        self.total_number[i] = np.sum(np.sum(self.next_density))
        self.total_volume[i] = np.sum(np.sum(self.next_density, axis=1)
        self.total_charge[i] = np.sum(np.sum(self.next_density, axis=0)*self.grid_data.qpivots)

        self.nano_history[i] = np.copy(self.next_density)
        self.krate_history[i] = np.copy(self.birth2d)
        self.sgrate_history[i] = np.copy(self.gsurface_growth)

    def post_processing(self):
        """ Additional calculations
        """
        self.ddens = [np.sum(d, axis=1) for d in self.nano_history]
        self.cdens = [np.sum(d, axis=0) for d in self.nano_history]

    def particle_peak(self):
        """ Computes the maximum peak for size distribution
        """
        self.maxparticle = [find_peaks(d)[0][-1]d in self.ddens]
        self.peak_diameter = savgol_filter(self.grid_data.dpivots[self.maxparticle], 201, 3)

class NanoData():
    """ Open and store the nano data
    """
    def __init__(self, h5nanoprefix, defpath):
        # Read nano file
        self.nanofname = defpath + h5nanoprefix + '.h5'
        self.nanofile = h5py.File(self.nanofname, 'r')

        # density group
        self.gdensity = self.nanofile.get("Density")
        self.result = np.array(self.gdensity.get("density"))

        self.initial_density = np.array(self.gdensity.get("initial_density")).T

        self.ddens = np.sum(self.result, axis=0)
        self.cdens = np.sum(self.result, axis=1)


    def close(self):
        """ Close nanofile
        """
        self.nanofile.close()

def nanoparticle_growth(growth_data, grid_data, surface_growth):
    """ Compute coagulation and surface growth
    """
    growth_data.death2d.fill(0.0)
    coag.parallel_death(grid_data.death_larray, grid_data.death_qarray,
                        grid_data.death_marray, grid_data.death_parray,
                        grid_data.death_array,
                        growth_data.past_density, growth_data.death2d)

    growth_data.birth2d.fill(0.0)
    coag.parallel_birth(grid_data.birth_larray, grid_data.birth_qarray,
                        grid_data.birth_marray, grid_data.birth_parray,
                        grid_data.birth_narray, grid_data.birth_rarray,
                        grid_data.birth_array,
                        growth_data.past_density, growth_data.birth2d)

    growth_data.gsurface_growth.fill(0.0)
    surface_growth.compute_sgrowth(growth_data.past_density,
                                   growth_data.gsurface_growth)

    growth_data.advance()
    growth_data.update_density()
