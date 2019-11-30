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
the plasma.
"""

__author__ = "Benjamin Santos"
__copyright__ = "Copyright 2019"
__credits__ = ["Benjamin Santos"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Benjamin Santos"
__email__ = "caos21@gmail.com"
__status__ = "Beta"

import numpy as np
import scipy.constants as const
from scipy.integrate import solve_ivp

PI = const.pi
KE = 1.0/(4.0*PI*const.epsilon_0)
INVKE = 1.0/KE
KB = const.Boltzmann
QE = const.elementary_charge
ME = const.electron_mass

def coulomb_floatpotential(qcharge, radius):
    """ Floating potential
    """
    return KE*qcharge/radius

def particle_potenergy(radius, zcharge):
    """ Nanoparticle potential energy
    """
    return -(KE*zcharge*QE**2)/radius

def tunnel(rtaff, radius, zcharge):
    """ Tunneling probability
    """
    prefac1 = -2./const.hbar
    prefac2 = np.sqrt(2.*ME*particle_potenergy(rtaff, zcharge))
    return np.exp(prefac1*prefac2*(rtaff*np.arccos(np.sqrt(radius/rtaff))
                                   -np.sqrt(radius*(rtaff-radius))))

class TunnelFrequency:
    """ Computes electron tunnel frequency
    """
    def __init__(self, plasmasystem):
        self.psys = plasmasystem
        self.eaffinity = 4.05*const.elementary_charge

    def __call__(self, zcharge, radius):
        return self.ptunnel(zcharge, radius)

    def rt_affinity(self, radius, zcharge):
        """ Computes rt_affinity to particle to escape
        """
        ainfinity = self.eaffinity
        ainf = ainfinity * INVKE/QE**2

        rtaff = zcharge/(zcharge/radius + ainf - (5.0/(8.0*radius)))

        if np.isscalar(rtaff):
            if rtaff < 0:
                return 1000000.0
        else:
            rtaff[rtaff < 0] = 1000000.0
        return rtaff

    def ptunnel(self, zcharge, radius):
        """ Tunnel frequency
        """
        prefac1 = (-zcharge)*np.sqrt(2.*const.Boltzmann*self.psys.temperature/ME)*(0.5/radius)
        rtaff = self.rt_affinity(radius, zcharge)
        return prefac1*tunnel(rtaff, radius, zcharge)

class CollisionFrequency:
    """ Stores and computes collision frequencies
    """
    def __init__(self, plasmasystem, grid_data):
        self.psys = plasmasystem
        self.gdata = grid_data
        self.tfrequency = TunnelFrequency(self.psys)

        self.rmesh, self.qmesh = np.meshgrid(self.gdata.dpivots*0.5e-9,
                                             self.gdata.qpivots*QE, indexing='ij')
        self.rmesh, self.zmesh = np.meshgrid(self.gdata.dpivots*0.5e-9,
                                             self.gdata.qpivots, indexing='ij')
        self.rmesh2 = self.rmesh**2

        self.phid = coulomb_floatpotential(self.qmesh, self.rmesh)

        self.ion_velocity = 0.0

    def compute_collisionfreq(self, energy, edensity, idensity, efreq, ifreq, tfreq):
        """ Compute collision frequencies OML theory and Tunnel frequency
        """
        kte = (2.0/3.0)*energy*QE

        efreqfactor = 4.0 * PI * edensity * np.sqrt(kte/(2.0*PI*ME))

        ion_energy_from_temperature = (3.0/2.0) * KB * self.psys.ion_temperature

        ion_energy = (ion_energy_from_temperature
                      + 0.5*self.psys.armass*self.ion_velocity*self.ion_velocity)
        kti = (2.0/3.0)*ion_energy

        ifreqfactor = 4.0 * PI * idensity * np.sqrt(kti/(2.0*PI*self.psys.armass))

        efreq.fill(0)
        ifreq.fill(0)
        tfreq.fill(0)

        gdata = self.gdata
        rmesh2 = self.rmesh2
        phid = self.phid

        efreq[:, gdata.qpivots < 0] = (efreqfactor * rmesh2[:, gdata.qpivots < 0]
                                       * np.exp(QE*phid[:, gdata.qpivots < 0]/kte))

        efreq[:, gdata.qpivots >= 0] = (efreqfactor * rmesh2[:, gdata.qpivots >= 0]
                                        * (1.0 + QE*phid[:, gdata.qpivots >= 0]/kte))

        ifreq[:, gdata.qpivots <= 0] = (ifreqfactor * rmesh2[:, gdata.qpivots <= 0]
                                        * (1.0 - QE*phid[:, gdata.qpivots <= 0]/kti))

        ifreq[:, gdata.qpivots > 0] = (ifreqfactor * rmesh2[:, gdata.qpivots > 0]
                                       * np.exp(-QE*phid[:, gdata.qpivots > 0]/kti))

        for i, diam in enumerate(gdata.dpivots):
            for j, zcharge in enumerate(gdata.qpivots[gdata.qpivots < 0]):
                tfreq[i][j] = self.tfrequency(zcharge, 0.5e-9*diam)

        for i, diam in enumerate(gdata.dpivots):
            for j, zcharge in enumerate(gdata.qpivots[gdata.qpivots < 0]):
                if (tfreq[i][j] > 1e6*ifreq[i][j]) and (ifreq[i][j] > efreq[i][j]):
                    tfreq[i][j] = 1e6*ifreq[i][j]

class Charging:
    """ Compute nanoparticle charging rate
    """
    def __init__(self, collision_frequency, grid_data):
        """
        """
        self.coll = collision_frequency
        self.grid_data = grid_data
        self.nvols = self.grid_data.nvols
        self.nchrgs = self.grid_data.nchrgs
        self.efreq = np.zeros((self.nvols, self.nchrgs))
        self.ifreq = np.zeros((self.nvols, self.nchrgs))
        self.tfreq = np.zeros((self.nvols, self.nchrgs))

    def compute_freqs(self, energy, edensity, idensity):
        """ Compute frequencies
        """
        self.coll.compute_collisionfreq(energy, edensity, idensity,
                                        self.efreq, self.ifreq, self.tfreq)

def compute_plasmacharging(time, delta_t, grid_data, pchem,
                           growth_data, charging, plasma_sys):
    """ Solve the plasma densities
    """
    with_tunnel = plasma_sys.with_tunnel
    nel = pchem.past_plasmadensity[0]
    nar = pchem.past_plasmadensity[1]
    npdensity = growth_data.next_density
    ion_loss = np.sum(npdensity*charging.ifreq)/nar
    electron_loss = np.sum(npdensity*charging.efreq)/nel
    energy_loss = np.sum(charging.coll.phid*npdensity*charging.efreq)/nel
    tunnel_gain = with_tunnel*np.sum(npdensity*charging.tfreq)
    energy_gain = with_tunnel*np.sum(charging.coll.phid*npdensity*charging.tfreq)

    pchem.density_sourcedrain = np.array([electron_loss, ion_loss, 0.0, 0.0,
                                          tunnel_gain, energy_gain, energy_loss])

    nano_qdens = np.sum(npdensity*grid_data.qpivots)
    pchem.nano_qdens = nano_qdens

    nano_qdens_rate = np.sum(growth_data.qrate2d*grid_data.qpivots)
    pchem.nano_qdens_rate = nano_qdens_rate

    plasma_sys = pchem.get_system()

    sol = solve_ivp(plasma_sys, [time, time+delta_t], pchem.past_plasmadensity,
                    method='BDF', dense_output=False, t_eval=[time, time+delta_t])

    pchem.next_plasmadensity = np.nan_to_num(sol.y.T[-1])

    # quasineutrality
    pchem.next_plasmadensity[3] = (pchem.next_plasmadensity[0]-pchem.nano_qdens
                                   -pchem.next_plasmadensity[1])
