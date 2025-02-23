import awkward as ak
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import hist
import os
from hist import Hist
import shutil

hep.style.use("CMS")
palette = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]

# shutil.rmtree('./plots', ignore_errors=True)#delete the plots directory

def calculate_binning(bin):
    width = bin[1] - bin[0]
    center = (bin[:-1] + bin[1:]) / 2
    return width, center

n_bins = 40
# Define pT binning
x_low = 0
x_high = 5000
binning = np.linspace(x_low, x_high, n_bins + 1)
width, center = calculate_binning(binning)

# Define eta binning
eta_x_low = -2.5
eta_x_high = 2.5
binning_eta = np.linspace(eta_x_low, eta_x_high, n_bins + 1)
width_eta, center_eta = calculate_binning(binning_eta)

# Define cosThetaStar binning
costhetastar_x_low = -1
costhetastar_x_high = 1
binning_costhetastar = np.linspace(costhetastar_x_low, costhetastar_x_high, n_bins + 1)
width_costhetastar, center_costhetastar = calculate_binning(binning_costhetastar)

# Define mass binning
mass_x_low = 0
mass_x_high = 7000
binning_mass = np.linspace(mass_x_low, mass_x_high, n_bins + 1)
width_mass, center_mass = calculate_binning(binning_mass)

Path('./plots/pt/').mkdir(exist_ok=True, parents=True)
Path('./plots/eta/').mkdir(exist_ok=True, parents=True)
Path('./plots/thetastar/').mkdir(exist_ok=True, parents=True)
Path('./plots/mass/').mkdir(exist_ok=True, parents=True)

processes = [f for f in os.listdir("./NTuples") if os.path.isdir(os.path.join("./NTuples", f))]

# Plot pT for EB photons
for process in processes:
    diphotons = ak.from_parquet(f'./NTuples/{process}/')

    EB_photons = diphotons[diphotons.lead_isScEtaEB & diphotons.sublead_isScEtaEB]

    EB_lead_photon_pt_hist = np.histogram(EB_photons.lead_pt, bins=binning)[0]
    EB_sublead_photon_pt_hist = np.histogram(EB_photons.sublead_pt, bins=binning)[0]

    plt.hist(center, bins=binning, weights=EB_lead_photon_pt_hist, histtype='step', label='lead', linewidth=3)
    plt.hist(center, bins=binning, weights=EB_sublead_photon_pt_hist, histtype='step', label='sublead', linewidth=3)

    hep.cms.label("Preliminary",loc=1,com=13.6)
    plt.yscale('log')
    plt.xlabel('Photon_$p_T$ (GeV)')
    plt.ylabel('count')
    plt.legend()
    plt.xlim(x_low-100, x_high)
    plt.ylim(bottom=0,top = 1e6)
    plt.title(f'{process}_EB',fontsize=20, loc = 'left', pad= 35)
    plt.savefig(f'./plots/pt/{process}_EB.png')
    plt.close()

# Plot pT for EE photons
for process in processes:
    diphotons = ak.from_parquet(f'./NTuples/{process}/')

    EE_photons = diphotons[diphotons.lead_isScEtaEE & diphotons.sublead_isScEtaEE]

    EE_lead_photon_pt_hist = np.histogram(EE_photons.lead_pt, bins=binning)[0]
    EE_sublead_photon_pt_hist = np.histogram(EE_photons.sublead_pt, bins=binning)[0]

    plt.hist(center, bins=binning, weights=EE_lead_photon_pt_hist, histtype='step', label='lead', linewidth=3)
    plt.hist(center, bins=binning, weights=EE_sublead_photon_pt_hist, histtype='step', label='sublead', linewidth=3)

    hep.cms.label("Preliminary",loc=1,com=13.6)
    plt.yscale('log')
    plt.xlabel('Photon_$p_T$ (GeV)')
    plt.ylabel('count')
    plt.legend()
    plt.xlim(x_low-100, x_high)
    plt.ylim(bottom=0,top = 1e6)
    plt.title(f'{process}_EE',fontsize=20, loc = 'left', pad= 35)
    plt.savefig(f'./plots/pt/{process}_EE.png')
    plt.close()

# Plot eta for all photons
for process in processes:
    diphotons = ak.from_parquet(f'./NTuples/{process}/')

    EB_photons = diphotons[diphotons.lead_isScEtaEB & diphotons.sublead_isScEtaEB]
    EE_photons = diphotons[diphotons.lead_isScEtaEE & diphotons.sublead_isScEtaEE]

    EB_lead_photon_eta_hist = np.histogram(EB_photons.lead_eta, bins=binning_eta)[0]
    EB_sublead_photon_eta_hist = np.histogram(EB_photons.sublead_eta, bins=binning_eta)[0]
    EE_lead_photon_eta_hist = np.histogram(EE_photons.lead_eta, bins=binning_eta)[0]
    EE_sublead_photon_eta_hist = np.histogram(EE_photons.sublead_eta, bins=binning_eta)[0]

    plt.hist(center_eta, bins=binning_eta, weights=EB_lead_photon_eta_hist, histtype='step', label='lead_EB', linewidth=3)
    plt.hist(center_eta, bins=binning_eta, weights=EB_sublead_photon_eta_hist, histtype='step', label='sublead_EB', linewidth=3)
    plt.hist(center_eta, bins=binning_eta, weights=EE_lead_photon_eta_hist, histtype='step', label='lead_EE', linewidth=3)
    plt.hist(center_eta, bins=binning_eta, weights=EE_sublead_photon_eta_hist, histtype='step', label='sublead_EE', linewidth=3)

    hep.cms.label("Preliminary",loc=1,com=13.6)
    plt.yscale('log')
    plt.xlabel('Photon_${\eta}$')
    plt.ylabel('count')
    plt.legend()
    plt.title(f'{process}',fontsize=20, loc = 'left', pad= 35)
    plt.xlim(eta_x_low-0.25, eta_x_high+0.25)
    plt.ylim(bottom=0, top = 2*1e4)
    plt.savefig(f'./plots/eta/{process}.png')
    plt.close()

# Plot cosThetaStar for all photons
for process in processes:
    diphotons = ak.from_parquet(f'./NTuples/{process}/')

    vetoed_diphotons = diphotons[diphotons.lead_electronVeto & diphotons.sublead_electronVeto]
    selected_diphotons = vetoed_diphotons[(vetoed_diphotons.lead_pt > 50) & (vetoed_diphotons.sublead_pt > 50)]

    cosThetaStar_his = np.histogram(selected_diphotons.cosThetaStar, bins=binning_costhetastar)[0]

    plt.hist(center_costhetastar, bins=binning_costhetastar, weights=cosThetaStar_his, histtype='step', density = True, label='cosThetaStar', linewidth=3)

    hep.cms.label("Preliminary",loc=1,com=13.6)
    plt.xlabel(r'$\cos{\theta^*}$')
    plt.ylabel('Density')
    # plt.yscale('log')
    plt.title(f'{process}',fontsize=20, loc = 'left', pad= 35)
    plt.xlim(costhetastar_x_low-0.1, costhetastar_x_high+0.1)
    plt.ylim(bottom=0, top = 1)
    plt.savefig(f'./plots/thetastar/{process}.png')
    plt.close()

# Plot mass for all diphotons
for process in processes:
    diphotons = ak.from_parquet(f'./NTuples/{process}/')

    vetoed_diphotons = diphotons[diphotons.lead_electronVeto & diphotons.sublead_electronVeto]
    selected_diphotons = vetoed_diphotons[(vetoed_diphotons.lead_pt > 50) & (vetoed_diphotons.sublead_pt > 50)]

    mass_his = np.histogram(selected_diphotons.mass, bins=binning_mass)[0]

    plt.hist(center_mass, bins=binning_mass, weights=mass_his, histtype='step',density = True, label='mass', linewidth=3)

    hep.cms.label("Preliminary",loc=1,com=13.6)
    plt.yscale('log')
    plt.xlabel('$M_{\gamma\gamma}$')
    plt.ylabel('Density')
    plt.title(f'{process}',fontsize=20, loc = 'left', pad= 35)
    plt.xlim(mass_x_low, mass_x_high)
    plt.ylim(bottom=0, top = 1)
    plt.savefig(f'./plots/mass/{process}.png')
    plt.close()