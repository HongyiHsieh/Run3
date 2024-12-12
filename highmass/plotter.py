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

shutil.rmtree('./plots', ignore_errors=True)#delete the plots directory

n_bins = 40
x_low = 0
x_high = 5000
binning = np.linspace(x_low, x_high, n_bins + 1)
width = binning[1] - binning[0]
center = (binning[:-1] + binning[1:]) / 2

eta_x_low = -2.5
eta_x_high = 2.5
binning_eta = np.linspace(eta_x_low, eta_x_high, n_bins + 1)
width_eta = binning_eta[1] - binning_eta[0]
center_eta = (binning_eta[:-1] + binning_eta[1:]) / 2

Path('./plots/pt/').mkdir(exist_ok=True, parents=True)
Path('./plots/eta/').mkdir(exist_ok=True, parents=True)

processes = [f for f in os.listdir("./NTuples") if os.path.isdir(os.path.join("./NTuples", f))]

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