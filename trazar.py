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
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.animation as animation
import seaborn as sns
sns.set(font_scale=2)
# Customizations
mpl.rcParams['lines.linewidth'] = 2
mpl.rc('font', family='DejaVu Sans')
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica']
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [
    r'\usepackage[version=4]{mhchem}',
    r'\usepackage{siunitx}',
    r'\sisetup{detect-all}',
    r'\usepackage{helvet}',
    r'\usepackage{sansmath}',
    r'\sansmath',
    r'\usepackage{textgreek}']
mpl.rcParams['axes.formatter.useoffset'] = False

CURRENT_PALETTE = sns.color_palette('bright')
#swap colors
CURRENT_PALETTE[4], CURRENT_PALETTE[6] = CURRENT_PALETTE[6], CURRENT_PALETTE[4]
#set
sns.set_palette(CURRENT_PALETTE)

sns.set_style("whitegrid", {'axes.linewidth': '2', 'axes.edgecolor': '0.15',
                            "xtick.major.size": 8, "ytick.major.size": 8,
                            "xtick.minor.size": 4, "ytick.minor.size": 4,
                            "lines.linewidth":8})

LINESTYLES = ['-', '--', '-.', ':']

def plot_distro(pivots, data, labels=None, logx=True, title=u"Size distribution",
                axislabel=(r'Diameter $(\text{nm)}$', r'Density $(m^{{-3}})$'),
                savename="distro.png", figname="ddiam", ylim=None, logy=True,
                linestyles=None):
    """ Plot distribution
    """
    # Create the figure

    fig = plt.figure(figname, figsize=(12, 9))
    plt.title(title)
    plt.xlabel(axislabel[0])
    plt.ylabel(axislabel[1])

    if logx:
        plt.xscale('log')

    if logy:
        plt.yscale('log')

    plt.xlim((pivots[0], pivots[-1]))

    data_max = 0.0
    if isinstance(data, list):
        if linestyles is None:
            linestyles = LINESTYLES[:]
        for i, (distro, label) in enumerate(zip(data, labels)):
            plt.plot(pivots, distro, linewidth=3, label=label, linestyle=linestyles[i%4])
        data_max = data[0].max()
        plt.legend()
    else:
        plt.plot(pivots, data, linewidth=3)
        data_max = data.max()

    plt.ylim((1, data_max))

    ax1 = fig.gca()
    ax1.grid(b=True, which='minor', axis='x', color='k', linewidth=0.5)

    if ylim is not None:
        plt.ylim(ylim)
    plt.savefig(savename, bbox_inches='tight')
    plt.show()

def plot_plain(pivots, data, labels=None, logx=True, title=u"Size distribution",
               axislabel=(r'Diameter $(\text{nm)}$', r'Density $(m^{{-3}})$'),
               savename="plain.png", figname="ddiam", xlim=None, ylim=None, logy=True,
               linestyles=None):
    """ Plot distribution
    """
    # Create the figure

    fig = plt.figure(figname, figsize=(12, 9))
    plt.title(title)
    plt.xlabel(axislabel[0])
    plt.ylabel(axislabel[1])

    if logx:
        plt.xscale('log')

    if logy:
        plt.yscale('log')

    if isinstance(data, list):
        if linestyles is None:
            linestyles = LINESTYLES[:]
        for i, (distro, label) in enumerate(zip(data, labels)):
            plt.plot(pivots, distro, linewidth=3, label=label, linestyle=linestyles[i%4])
        #data_max = data[0].max()
        plt.legend()
    else:
        plt.plot(pivots, data, linewidth=3)

    ax1 = fig.gca()
    ax1.grid(b=True, which='minor', axis='x', color='k', linewidth=0.5)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.savefig(savename, bbox_inches='tight')

    plt.show()

def process_input(inputarg):
    """ Process input
    """
    if not isinstance(inputarg, list):
        inputarg = [inputarg]

    if isinstance(inputarg[0], str):
        data = []
        for file in inputarg:
            try:
                data.append(np.loadtxt(file))
            except OSError as err:
                print("Wrong filename, ", err)

        return data

    return inputarg

class PlotPanels:
    """ Plot planels
    """
    def __init__(self, moments, plasma, labels,
                 savefile, coagulation_alone=False):
        """
        """

        self.moments = process_input(moments)
        self.plasma = process_input(plasma)

        self.labels = labels
        self.savefile = savefile
        self.coagulation_alone = coagulation_alone

        self.lts = LINESTYLES
        self.time_f = 1.0

    def plot_comp(self, savefile):
        """ Plot to compare qdensity and nano total number density
        """
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 10.5))

        ax1.set_xlabel("Time (s)")
        ax1.set_xlim(0.0, 0.6)
        ax1.set_ylabel(r"Number density (cm$^{-3}$)")
        #ax1.set_yscale("log")
        for i, data in enumerate(self.moments):
            ax1.plot(data[:, 0], data[:, 1]*1e-6, ls=self.lts[i%(len(self.lts))], lw=3)
            ax1.plot(data[:, 0], -data[:, 3]*1e-6, ls=self.lts[i%(len(self.lts))], lw=3)
        ax1.set_ylim([0, 6e10])
        fig.savefig(savefile)
        plt.show()

    def plot(self):
        """ Plot panels
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,
                                                     sharex=True,
                                                     figsize=(14, 10.5))

        lflist = []
        for i, data in enumerate(self.moments):
            lfl, = ax1.plot(data[:, 0], data[:, 1]*1e-16, ls=self.lts[i%(len(self.lts))], lw=3)
            ax3.plot(data[:, 0], np.abs(data[:, 3])*1e-16, ls=self.lts[i%(len(self.lts))], lw=3)
            ax4.plot(self.plasma[i][:, 0], self.plasma[i][:, 1],
                     ls=self.lts[i%(len(self.lts))], lw=3)
            lflist.append(lfl)

            if not self.coagulation_alone:
                ax2.plot(data[:, 0], data[:, 2]*1e6, ls=self.lts[i%(len(self.lts))], lw=3)
            else:
                ax2.plot(data[:, 0], data[:, 2]*1e12, ls=self.lts[i%(len(self.lts))], lw=3)

        for i, (axs, text) in enumerate(zip([ax1, ax2, ax3, ax4], ['a', 'b', 'c', 'd'])):
            axs.text(0.5, 0.9, text, transform=axs.transAxes, fontsize=24)
            axs.grid(b=True, which='minor', axis='x', linewidth=1)
            axs.grid(b=True, which='minor', axis='y', linewidth=1)

        if not self.coagulation_alone:
            ax2.set_ylabel(r"Total volume ($10^{\text{-}6}$m$^3$/m$^3$)")
            ax2.set_ylim(bottom=0.0)
        else:
            ax2.set_ylabel(r"Total volume ($10^{\text{-}12}$m$^3$/m$^3$)")
            ax2.set_ylim(2.1, 2.4)

        ax1.set_xlim(0.0, self.time_f)
        ax1.set_ylabel(r"Number density $(10^{16}$m$^{\text{-}3})$")
        ax1.set_ylim(bottom=0.0)
        ax3.set_ylim(bottom=0.0)
        ax3.set_ylabel(r"Total charge ($-10^{16}$ e/m$^3$)")
        ax4.set_ylabel(r"Mean electron energy (eV)")
        ax3.set_xlabel("Time (s)")
        ax4.set_xlabel("Time (s)")

        fig.legend(tuple(lflist), self.labels, loc='upper center', ncol=5, labelspacing=0.)

        #plt.tight_layout()
        plt.subplots_adjust(wspace=0.35)
        fig.savefig(self.savefile)#, bbox_inches="tight")
        plt.show()

    def plot2(self):
        """ Plot panels
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,
                                                     sharex=True,
                                                     figsize=(14, 10.5))

        lflist = []
        #nelist = []
        ax3_2 = ax3.twinx()
        for i, data in enumerate(self.moments):
            lfl, = ax1.plot(data[:, 0], data[:, 1]*1e-16, ls=self.lts[i%(len(self.lts))], lw=3)
            ax3.plot(data[:, 0], np.abs(data[:, 3])*1e-16, ls=self.lts[i%(len(self.lts))], lw=3)
            color3_2 = CURRENT_PALETTE[7]
            if i > 3:
                color3_2 = 'k'
            ax3_2.plot(self.plasma[i][:, 0], self.plasma[i][:, 2],
                       ls=self.lts[i%(len(self.lts))], lw=3,
                       color=color3_2, label=r'$n_e$ - '+self.labels[i])
            ax4.plot(self.plasma[i][:, 0], self.plasma[i][:, 1],
                     ls=self.lts[i%(len(self.lts))], lw=3)
            lflist.append(lfl)
            #nelist.append(ne)

            if not self.coagulation_alone:
                ax2.plot(data[:, 0], data[:, 2]*1e6, ls=self.lts[i%(len(self.lts))], lw=3)
            else:
                ax2.plot(data[:, 0], data[:, 2]*1e12, ls=self.lts[i%(len(self.lts))], lw=3)

        for i, (axs, text) in enumerate(zip([ax1, ax2, ax3, ax4], ['a', 'b', 'c', 'd'])):
            axs.text(0.5, 0.9, text, transform=axs.transAxes, fontsize=24)
            axs.grid(b=True, which='minor', axis='x', linewidth=1)
            axs.grid(b=True, which='minor', axis='y', linewidth=1)
            axs.set_ylim(bottom=0.0)

        if not self.coagulation_alone:
            ax2.set_ylabel(r"Total volume ($10^{\text{-}6}$m$^3$/m$^3$)")
            ax3_2.set_yscale('log')
            ax4.set_ylim(bottom=1.0)
            ax3.set_xlim(0.0, self.time_f)
            ax3_2.set_xlim(0.0, self.time_f)
            ax3_2.legend(fontsize='x-large', markerscale=1.5, loc='lower left')
        else:
            ax2.set_ylabel(r"Total volume ($10^{\text{-}12}$m$^3$/m$^3$)")
            ax2.set_ylim(2.1, 2.4)


        ax1.set_xlim(0.0, self.time_f)
        #ax3_2.set_ylim(bottom=10.0)

        #ax3.set_ylim(ax3_2.get_ylim())
        ax1.set_ylabel(r"Number density $(10^{16}$m$^{\text{-}3})$")
        ax3.set_ylabel(r"Total charge ($-10^{16}$ e/m$^3$)")
        ax3_2.set_ylabel(r"Electron density (m$^{-3}$)")
        ax4.set_ylabel(r"Mean electron energy (eV)")
        ax3.set_xlabel("Time (s)")
        ax4.set_xlabel("Time (s)")

        fig.legend(tuple(lflist), self.labels, loc='upper center', ncol=5, labelspacing=0.)

        #plt.tight_layout()
        plt.subplots_adjust(wspace=0.46)
        fig.savefig(self.savefile)#, bbox_inches="tight")
        plt.show()

class DistroAnimated():
    """ Plots the nanoparticle distribution as a function of time
    """
    def __init__(self, plotdistros, potential):
        """
        """
        self.pdistro = plotdistros
        self.potential = potential
        self.fig = plt.figure(figsize=(14, 14))

        self.gspec = gridspec.GridSpec(2, 3,
                                       width_ratios=[1, 3.5, 0.2],
                                       height_ratios=[3.5, 1])

        self.axcharges = plt.subplot(self.gspec[0])
        self.axsizes = plt.subplot(self.gspec[4])
        self.axdistro = plt.subplot(self.gspec[1], sharey=self.axcharges,
                                    sharex=self.axsizes)
        self.axcbar = plt.subplot(self.gspec[2])

        plt.subplots_adjust(wspace=0.05, hspace=0.04)

        self.psizes, = self.axsizes.plot([], [], label=self.potential, lw=3)
        self.pcharges, = self.axcharges.plot([], [], label=self.potential, lw=3)

        self.cspmap = None
        self.cmap = None

        self.time_template = 'time = %.2fs'
        self.time_text = self.axdistro.text(0.7, 0.2, '',
                                            transform=self.axdistro.transAxes)
        self.levelsf = None

    def init(self):
        """ Prepare for animation
        """
        axsizes_ = self.axsizes
        axcharges_ = self.axcharges
        axdistro_ = self.axdistro
        pds = self.pdistro

        axsizes_.set_xlabel('Diameter (nm)')
        axsizes_.set_ylabel(r'Density (m$^{-3}$)')
        axsizes_.set_xscale('log')
        axsizes_.set_yscale('log')
        axsizes_.set_ylim([100, 10e16])
        axsizes_.set_xlim([pds.dpivots[0], pds.dpivots[-1]])
        axsizes_.grid(b=True, which='minor', axis='x', linewidth=0.5)

        axdistro_.grid(b=True, which='minor', axis='x', linewidth=0.5)
        axdistro_.grid(b=True, which='minor', axis='y', linewidth=0.5)

        #pcharges_.set_title('Charge distribution')
        axcharges_.set_ylabel('Charge (e)')
        axcharges_.set_xlabel(r'Density (e/m$^{3}$)')
        axcharges_.set_xscale('log')
        axcharges_.set_xlim([100, 50e16])
        axcharges_.set_ylim([pds.qpivots[0]+20, pds.qpivots[-1]])
        axcharges_.xaxis.set_label_position("top")
        axcharges_.xaxis.tick_top()

        minor_ticks = np.arange(pds.qpivots[0]+20, pds.qpivots[-1], 2)
        axcharges_.set_yticks(minor_ticks, minor=True)
        axcharges_.grid(b=True, which='minor', axis='y', linewidth=0.5)


        plt.setp(axdistro_.get_xticklabels(), visible=False)
        plt.setp(axdistro_.get_yticklabels(), visible=False)

        self.levelsf = np.arange(2, 18, 2)

        self.time_text.set_text('')

        self.psizes.set_data([], [])
        self.pcharges.set_data([], [])

        self.cspmap = []
        self.cmap = []

        return self.psizes, self.pcharges, self.cspmap, self.cmap, self.time_text

    def update(self, i):
        """ Update frame
        """
        pds = self.pdistro
        self.psizes.set_data(pds.dpivots, pds.sizedistros[self.potential][1][i])
        self.pcharges.set_data(pds.chargedistros[self.potential][1][i], pds.qpivots)

        result = pds.fulldistros[self.potential][1][i]
        res_gt_0 = result > 0
        log10res = np.zeros_like(result)
        log10res[res_gt_0] = np.log10(result[res_gt_0])

        # remove all contours
        self.axdistro.collections = []

        # update contours
        self.cmap = self.axdistro.contour(pds.dmesh, pds.qmesh, log10res,
                                          levels=self.levelsf, origin='lower',
                                          linewidths=1.5, colors='k')
        self.cspmap = self.axdistro.contourf(pds.dmesh, pds.qmesh, log10res,
                                             levels=self.levelsf, origin='lower',
                                             cmap=cm.magma_r)#, extent=self.extents_linear)
        plt.colorbar(self.cspmap, cax=self.axcbar)

        self.time_text.set_text(self.time_template%(pds.chargedistros[self.potential][0][i]))

        return self.psizes, self.pcharges, self.cspmap, self.cmap, self.time_text

    def animation(self, step=2):
        """ Returns the animation
        """
        return animation.FuncAnimation(self.fig, self.update, frames=np.arange(0, 100, step),
                                       init_func=self.init, blit=False)
class PlotDistros():
    """ Plots the distributions
    """
    def __init__(self, h5nanoprefix, h5gridprefix, labels):

        self.defpath = r'/mnt/data/ben/ndust/data/'

        self.nanofiles = [self.defpath + h5n + '.h5' for h5n in h5nanoprefix]
        # Read nano file
        self.nanofile = h5py.File(self.nanofiles[0], 'r')

        # Read grid file
        self.h5gridprefix = h5gridprefix
        self.h5gridprefix = self.defpath + self.h5gridprefix
        self.gridfname = self.h5gridprefix + '.h5'
        self.gridfile = h5py.File(self.gridfname, 'r')

        self.labels = labels

        ## group volumes
        self.gvols = self.gridfile.get("Volume_sections")
        self.vifaces = np.array(self.gvols.get("Interfaces"))

        ## interfaces in diameters nm
        self.vifaces_diam = np.power(6.0*self.vifaces/np.pi, 1.0/3.0)*1E9

        ## WARNING diameter pivots in nanometres
        self.vpivots = np.array(self.gvols.get("Volumes"))

        ## pivots in diameters
        self.dpivots = np.array(self.gvols.get("Diameters"))*1E9

        ## group charges
        self.gchgs = self.gridfile.get("Charge_sections")

        self.qpivots = np.array(self.gchgs.get("Charges"))

        self.dmesh, self.qmesh = np.meshgrid(self.dpivots, self.qpivots)

        # density group
        self.gdensity = self.nanofile.get("Density")
        self.result = np.array(self.gdensity.get("density"))

        self.ddens = np.sum(self.result, axis=0)
        self.cdens = np.sum(self.result, axis=1)

        self.fulldistros = dict()
        self.sizedistros = dict()
        self.chargedistros = dict()
        for i, file in enumerate(self.nanofiles):
            h5data = h5py.File(file, 'r')
            gdensity = h5data.get("Density")
            fulldistro = []
            sizedistro = []
            totalnumber = []
            chargedistro = []

            times = []
            for name, dset in gdensity.items():
                time = gdensity.get(name).attrs.get('time')
                if time:
                    fulldistro.append(dset)
                    sizedistro.append(np.sum(dset, axis=0))
                    totalnumber.append(np.sum(dset))
                    chargedistro.append(np.sum(dset, axis=1))
                    times.append(time)

            self.fulldistros[labels[i]] = [times, np.asarray(fulldistro)]
            self.sizedistros[labels[i]] = [times, sizedistro, totalnumber]
            self.chargedistros[labels[i]] = [times, chargedistro]

            h5data.close()

        self.lts = LINESTYLES
        self.close()

    def plot_fcdistro(self, potential, savename):
        """ Plot the distribution contour surface
        """
        fig, ax1 = plt.subplots(figsize=(12, 9))

        ax1.set_xlabel('Diameter (nm)')
        ax1.set_xscale('log')
        ax1.set_xlim([self.dpivots[0], self.dpivots[-1]])

        ax1.grid(b=True, which='minor', axis='x', linewidth=0.5)
        ax1.grid(b=True, which='minor', axis='y', linewidth=0.5)

        ax1.set_ylabel('Charge (e)')
        ax1.set_ylim([-40, self.qpivots[-1]])

        minor_ticks = np.arange(-40, self.qpivots[-1], 2)
        ax1.set_yticks(minor_ticks, minor=True)

        levelsf = np.arange(2, 18, 2)

        result = self.fulldistros[potential][1][-1]
        res_gt_0 = result > 0
        log10res = np.zeros_like(result)
        log10res[res_gt_0] = np.log10(result[res_gt_0])

        cspmap = ax1.contourf(self.dmesh, self.qmesh, log10res, levels=levelsf,
                              origin='lower', cmap=cm.magma_r)#, extent=self.extents_linear)

        csl = ax1.contour(self.dmesh, self.qmesh, log10res, levels=levelsf,
                          origin='lower', linewidths=1.5, colors='k')
        cbar = plt.colorbar(cspmap)
        cbar.ax.set_ylabel(r'$\log N_{ik}$')
        cbar.ax.tick_params()
        cbar.add_lines(csl)

        #plt.tight_layout()
        plt.savefig(savename, bbox_inches='tight')
        plt.show()

    def plot_fulldistro(self, potential):
        """ Plot the distribution and subplots in axes
        """
        fig = plt.figure(figsize=(14, 14))
        fig.suptitle('Distribution', size=24)
        gs_ = gridspec.GridSpec(2, 3,
                                width_ratios=[1, 3.5, 0.2],
                                height_ratios=[3.5, 1])

        pcharges_ = plt.subplot(gs_[0])
        psizes_ = plt.subplot(gs_[4])
        pdistro_ = plt.subplot(gs_[1], sharey=pcharges_, sharex=psizes_)
        cbar_ = plt.subplot(gs_[2])

        plt.subplots_adjust(wspace=0.05, hspace=0.04)


        psizes_.set_xlabel('Diameter (nm)')
        psizes_.set_ylabel(r'Density (m$^{-3}$)')
        psizes_.set_xscale('log')
        psizes_.set_yscale('log')
        psizes_.set_ylim([100, 10e16])
        psizes_.set_xlim([self.dpivots[0], self.dpivots[-1]])
        psizes_.grid(b=True, which='minor', axis='x', linewidth=0.5)

        pdistro_.grid(b=True, which='minor', axis='x', linewidth=0.5)
        pdistro_.grid(b=True, which='minor', axis='y', linewidth=0.5)

        pcharges_.set_ylabel('Charge (e)')
        pcharges_.set_ylabel(r'Density (e/m$^{3}$)')
        pcharges_.set_xscale('log')
        pcharges_.set_xlim([100, 10e16])
        pcharges_.set_ylim([self.qpivots[0], -40])
        pcharges_.xaxis.set_label_position("top")
        pcharges_.xaxis.tick_top()

        minor_ticks = np.arange(self.qpivots[0], self.qpivots[-1], 2)
        pcharges_.set_yticks(minor_ticks, minor=True)
        pcharges_.grid(b=True, which='minor', axis='y', linewidth=0.5)

        psizes_.plot(self.dpivots, self.sizedistros[potential][1][-1],
                     color=CURRENT_PALETTE[0])
        pcharges_.plot(self.chargedistros[potential][1][-1], self.qpivots,
                       color=CURRENT_PALETTE[1])

        levelsf = np.arange(2, 18, 2)


        result = self.fulldistros[potential][1][-1]
        res_gt_0 = result > 0
        log10res = np.zeros_like(result)
        log10res[res_gt_0] = np.log10(result[res_gt_0])

        cspmap_ = pdistro_.contourf(self.dmesh, self.qmesh, log10res, levels=levelsf,
                                    origin='lower', cmap=cm.magma_r)#, extent=self.extents_linear)

        pdistro_.contour(self.dmesh, self.qmesh, log10res, levels=levelsf,
                         origin='lower', linewidths=1.5, colors='k')

        # hide tick labels in contoursf
        plt.setp(pdistro_.get_xticklabels(), visible=False)
        plt.setp(pdistro_.get_yticklabels(), visible=False)

        plt.colorbar(cspmap_, cax=cbar_)

        plt.show()

    def plot_2times(self, time1, time2, labels, savefile=None):
        """ Plot distros at time1 and time2
        """
        fig = plt.figure('Densities', figsize=(16, 7))

        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, sharey=ax1)

        ax1.text(0.1, 0.1, 'a', transform=ax1.transAxes, fontsize=24)
        ax2.text(0.1, 0.1, 'b', transform=ax2.transAxes, fontsize=24)

        ax1.set_xlabel('Diameter (nm)')
        ax1.set_ylabel(r'Density (m$^{-3}$)')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_ylim([10, 10e16])
        ax1.set_xlim([self.dpivots[0], self.dpivots[-1]])
        ax1.grid(b=True, which='minor', axis='x', linewidth=1)

        ax2.set_xlabel('Charge (e)')
        ax2.set_yscale('log')
        ax2.set_ylim([10, 10e16])
        ax2.set_xlim([-50, self.qpivots[-1]])
        ax2.grid(b=True)

        plt.setp(ax2.get_yticklabels(), visible=False)

        for j, name in enumerate(labels):
            ax1.plot(self.dpivots, self.sizedistros[name][1][time1], lw=3)
            ax2.plot(self.qpivots, self.chargedistros[name][1][time1], label=name, lw=3)

        for j, name in enumerate(labels):
            ax1.plot(self.dpivots, self.sizedistros[name][1][time2],
                     color=CURRENT_PALETTE[j], ls='-.', lw=3)
            ax2.plot(self.qpivots, self.chargedistros[name][1][time2],
                     color=CURRENT_PALETTE[j], ls='-.', lw=3)

        ax2.legend()

        print('index t1={}, time={}'.format(time1, self.sizedistros[labels[0]][0][time1]))

        minor_ticks = np.arange(-50, self.qpivots[-1], 5)
        ax2.set_xticks(minor_ticks, minor=True)
        ax2.grid(b=True, which='minor', axis='x', linewidth=1)

        plt.subplots_adjust(wspace=0.1)#, hspace=0.04)

        plt.tight_layout()
        if savefile:
            plt.savefig(savefile, bbox_inches='tight')
        plt.plot()

    def close(self):
        """ Close files
        """
        self.nanofile.close()
        self.gridfile.close()
