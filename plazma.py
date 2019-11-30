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

from collections import namedtuple
import numpy as np
import scipy.constants as const

import trazar as tzr

PI = const.pi
KE = 1.0/(4.0*PI*const.epsilon_0)
INVKE = 1.0/KE
KB = const.Boltzmann
QE = const.elementary_charge
ME = const.electron_mass

PlasmaSystem = namedtuple('System',
                          ['length',
                           'radius',
                           'temperature',
                           'ion_temperature',
                           'pressure_torr',
                           'arsih4_ratio',
                           'armass',
                           'sih4mass',
                           'power',
                           'with_tunnel'])

def constant_rate(energy, avar, bvar, cvar):
    """ Returns a constant rate a
    """
    return avar*np.ones_like(energy)

def arrhenius_rate(energy, avar, bvar, cvar):
    """ Returns the Arrhenius rate
    """
    return avar * np.power(energy, cvar) * np.exp(-bvar/energy)

def a1expb_rate(energy, avar, bvar, cvar):
    """ Returns a1expb rate
    """
    return avar * (1.0 - np.exp(-bvar*energy))

class RateSpec:
    """ Defines a rate
    """
    def __init__(self, rate_function=None, avar=0.0, bvar=0.0, cvar=0.0, name=""):
        self.rate_function = rate_function
        self.avar = avar
        self.bvar = bvar
        self.cvar = cvar
        self.name = name

    def __call__(self, energy):
        """ Returns the rate at mean electron energy value
        """
        return self.rate_function(energy, self.avar, self.bvar, self.cvar)

class RatesMap:
    """ Returns a dict of rates
    """
    def __init__(self, rates_dict):
        """
        """
        self.rates_dict = rates_dict
        self.rates_map = dict()

    def get_ratesmap(self):
        """ Get the rates map
        """
        for k, var in self.rates_dict.items():
            if var[0] == "a1expb":
                self.rates_map[k] = RateSpec(a1expb_rate, var[1], var[2], var[3], k)
            if var[0] == "arrhenius":
                self.rates_map[k] = RateSpec(arrhenius_rate, var[1], var[2], var[3], k)
            if var[0] == "constant":
                self.rates_map[k] = RateSpec(constant_rate, var[1], var[2], var[3], k)
        return self.rates_map

    def plot_rates(self, energy, savename="figx.eps"):
        """ Plot the rates
        """
        rates, labels = [], []
        for k, var in self.rates_map.items():
            rates.append(var(energy))
            labels.append(var.name)
        tzr.plot_plain(energy, rates, title="Rates",
                       axislabel=["Time (s)", r"Rate (m$^{-3}$s$^{-1}$)"],
                       logx=False, logy=True, labels=labels,
                       ylim=[1e-18, 1e-12], savename=savename)

class PlasmaChem():
    """ Plasma model
    """
    def __init__(self, rates_map, plasmasystem):
        self.rates_map = rates_map
        self.plasmasystem = plasmasystem

        self.electron_density = 1.0

        self.nano_qdens = 0.0
        self.nano_qdens_rate = 0.0
        self.kbtg = KB * self.plasmasystem.temperature
        self.ion_kbtg = KB * self.plasmasystem.ion_temperature
        self.pressure = 133.32237 * self.plasmasystem.pressure_torr
        self.reactor_volume = (self.plasmasystem.length*PI*self.plasmasystem.radius
                               *self.plasmasystem.radius)
        self.reactor_area = self.plasmasystem.length*2.0*PI*self.plasmasystem.radius
        self.ratio_av = self.reactor_area / self.reactor_volume
        self.gas_dens = self.pressure / self.kbtg
        self.nar = self.plasmasystem.arsih4_ratio * self.gas_dens
        self.nsih4 = (1.0-self.plasmasystem.arsih4_ratio) * self.gas_dens
        self.vth_ar = self.thermal_velocity(self.plasmasystem.armass)
        self.vth_sih4 = self.thermal_velocity(self.plasmasystem.sih4mass)

        self.flux_sih3 = self.flux_neutrals(self.plasmasystem.sih4mass)
        self.flux_sih2 = self.flux_neutrals(self.plasmasystem.sih4mass)
        self.flux_ar = self.flux_neutrals(self.plasmasystem.armass)

        ## From Lieberman pag 80 (117)
        self.lambdai = 1. / (330 * self.plasmasystem.pressure_torr)

        self.flux_arp = self.flux_ions(self.plasmasystem.armass, self.lambdai)
        self.flux_sih3p = self.flux_ions(self.plasmasystem.sih4mass, 2.9e-3)

        ## peak voltage
        self.vsheath = 0.25*100.0

        self.density_sourcedrain = np.zeros(7)
        self.past_plasmadensity = np.ones(7)
        self.next_plasmadensity = np.zeros(7)

    def thermal_velocity(self, mass):
        """ computes the thermal velocity
        """
        return np.sqrt(2.0*self.kbtg/mass)

    def diffusion_neutrals(self, mass, lambdax=3.5*1e-3):
        """ computes the diffusion coefficient for neutrals
        """
        return self.kbtg*lambdax/(mass*self.thermal_velocity(mass))

    def center2edge_neutrals(self, mass):
        """ center to edge ratio for neutrals
        """
        pfcn = (1.0 + (self.plasmasystem.length/2.0) * self.thermal_velocity(mass)
                / (4.0*self.diffusion_neutrals(mass)))
        return 1.0/pfcn

    def flux_neutrals(self, mass):
        """ computes the neutral flux
        """
        return 0.25 * self.center2edge_neutrals(mass) * self.thermal_velocity(mass)

    def bohm_velocity(self, mass):
        """ computes the Bohm velocity
        """
        return np.sqrt(self.ion_kbtg/mass)

    def center2edge_ions(self, lambdax):
        """ center to edge ratio for ions
        """
        pfcn = np.sqrt(3.0+(0.5*self.plasmasystem.length/lambdax))
        return 1.0/pfcn

    def flux_ions(self, mass, lambdax):
        """ computes the ion flux
        """
        return self.center2edge_ions(lambdax) * self.bohm_velocity(mass)

    def ion_velocity(self, mass):
        """ computes the ion velocity
        """
        return np.sqrt(8.0*self.ion_kbtg/(PI*mass))

    def get_system(self):
        """ returns the system of equations
        """
        return self.system

    def system(self, time, nvector):
        """ system of equations for the densities
        """
        nel = nvector[0]
        narp = nvector[1]
        narm = nvector[2]
        nsih3p = nvector[3]
        nsih3 = nvector[4]
        nsih2 = nvector[5]
        neps = nvector[6]

        energy = neps/nel

        kel = self.rates_map["R1:kel"](energy)
        kio = self.rates_map["R2:ki"](energy)
        kex = self.rates_map["R3:kex"](energy)
        kiarm = self.rates_map["R4:kiarm"](energy)
        kelsih4 = self.rates_map["R5:kelsih4"](energy)
        kdisih4 = self.rates_map["R6:kdisih4"](energy)
        kdsih3 = self.rates_map["R7:kdsih3"](energy)
        kdsih2 = self.rates_map["R8:kdsih2"](energy)
        kisih3 = self.rates_map["R9:kisih3"](energy)
        kv13 = self.rates_map["R10:kv13"](energy)
        kv24 = self.rates_map["R11:kv24"](energy)
        k12 = self.rates_map["R12:k12"](energy)
        k13 = self.rates_map["R13:k13"](energy)
        k14 = self.rates_map["R14:k14"](energy)
        k15 = self.rates_map["R15:k15"](energy)

        ekio = self.rates_map["R2:ki"].bvar
        ekex = self.rates_map["R3:kex"].bvar
        ekiarm = self.rates_map["R4:kiarm"].bvar
        ekdisih4 = self.rates_map["R6:kdisih4"].bvar
        ekdsih3 = self.rates_map["R7:kdsih3"].bvar
        ekdsih2 = self.rates_map["R8:kdsih2"].bvar
        ekisih3 = self.rates_map["R9:kisih3"].bvar
        ekv13 = self.rates_map["R10:kv13"].bvar
        ekv24 = self.rates_map["R11:kv24"].bvar

        nar = self.nar
        nsih4 = self.nsih4
        flux_arp = self.flux_arp
        flux_ar = self.flux_ar
        flux_sih3p = self.flux_sih3p
        flux_sih3 = self.flux_sih3
        flux_sih2 = self.flux_sih2
        ratio_av = self.ratio_av

        sourcedrain = self.density_sourcedrain
        with_tunnel = self.plasmasystem.with_tunnel

        nsih3p = nel - narp - self.nano_qdens

        dnel = (+kio*nar*nel
                + kiarm*nel*narm
                + kdisih4*nel*nsih4
                + kisih3*nel*nsih3
                - flux_arp*ratio_av*narp
                - flux_sih3p*ratio_av*nsih3p
                - sourcedrain[0]*nel
                + with_tunnel*sourcedrain[4])

        dnarp = (+kio*nar*nel
                 + kiarm*nel*narm
                 - flux_arp*ratio_av*narp
                 - sourcedrain[1]*narp)

        dnarm = (+ kex*nar*nel
                 - kiarm*narm*nel
                 - k12*narm*nsih4
                 - k13*narm*nsih4
                 - k14*narm*nsih3
                 - k15*narm*nsih2
                 - flux_ar*ratio_av*narm)

        dnsih3p = (+ kdisih4*nel*nsih4
                   + kisih3*nel*nsih3
                   - flux_sih3p*ratio_av*nsih3p)

        dnsih3 = (+ kdsih3*nel*nsih4
                  - kisih3*nel*nsih3
                  + k12*narm*nsih4
                  - k14*narm*nsih3
                  - flux_sih3*ratio_av*nsih3)

        dnsih2 = (+ kdsih2*nel*nsih4
                  + k13*narm*nsih4
                  + k14*narm*nsih3
                  - k15*narm*nsih2
                  - flux_sih2*ratio_av*nsih2)

        power = self.plasmasystem.power
        reactor_volume = self.reactor_volume
        vsheath = self.vsheath

        armass = self.plasmasystem.armass
        sih4mass = self.plasmasystem.sih4mass

        dneps = (power/reactor_volume
                 - ekio*kio*nar*nel
                 - ekex*kex*nar*nel
                 - ekiarm*kiarm*narm*nel
                 - (5./3.)*self.bohm_velocity(armass)*ratio_av*neps
                 - QE*vsheath*self.bohm_velocity(armass)*ratio_av*nel
                 - (5./3.)*self.bohm_velocity(sih4mass)*ratio_av*neps
                 - QE*vsheath*self.bohm_velocity(sih4mass)*ratio_av*nel
                 - 3.0*(ME/armass)*kel*neps*nar
                 - 3.0*(ME/sih4mass)*kelsih4*neps*nsih4
                 - ekisih3*kisih3*nel*nsih3
                 - ekdisih4*kdisih4*nel*nsih4
                 - ekdsih3*kdsih3*nel*nsih4
                 - ekdsih2*kdsih2*nel*nsih4
                 - ekv13*kv13*nel*nsih4
                 - ekv24*kv24*nel*nsih4
                 - sourcedrain[6]*nel
                 + with_tunnel*sourcedrain[5])

#        rdens = np.array([dnel, dnarp, dnarm, dnsih3p, dnsih3, dnsih2, dneps])
#        rdens[rdens<0.0] = 0.0
#        rdens[np.isnan(rdens)] = 0.0
#        return rdens
        return np.nan_to_num([dnel, dnarp, dnarm, dnsih3p, dnsih3, dnsih2, dneps],
                             copy=False)
