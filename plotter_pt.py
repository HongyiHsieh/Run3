# %%
import awkward as ak
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import hist
from hist import Hist
import os
import shutil

shutil.rmtree('./plots', ignore_errors=True)#delete the plots directory

hep.style.use("CMS")
palette = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]

# %%
parquetDir = "./parquet/"
#search for all parquet files in the directory
parquetFiles = [f for f in os.listdir(parquetDir) if f.endswith('.parquet')]
# set the number of bins
n_bins = 40
# Defintion of the pT binning
pt_x_low = 0
pt_x_high = 5000
binning_pt = np.linspace(pt_x_low, pt_x_high, n_bins + 1)
width_pt = binning_pt[1] - binning_pt[0]
center_pt = (binning_pt[:-1] + binning_pt[1:]) / 2
# Defintion of the eta binning
eta_x_low = -2.5
eta_x_high = 2.5
binning_eta = np.linspace(eta_x_low, eta_x_high, n_bins + 1)
width_eta = binning_eta[1] - binning_eta[0]
center_eta = (binning_eta[:-1] + binning_eta[1:]) / 2
# %%
# directory for output plots
pt_plots_dir = './plots/pt/'
Path(pt_plots_dir).mkdir(exist_ok=True, parents=True)

eta_plots_dir = './plots/eta/'
Path(eta_plots_dir).mkdir(exist_ok=True, parents=True)
# pT plot
for fileName in parquetFiles:
    # read the parquet file
    file_path = os.path.join(parquetDir, fileName)
    photon = ak.from_parquet(file_path)

    # %%
    # get the name of the file
    name = os.path.splitext(fileName)[0]
    part = name.split("_")
    if len(part) > 7:
        EEorEB = "_".join(name.split("_")[5:6])
        datasetname = "_".join(name.split("_")[:5])
        plt.title(f"{datasetname}",fontsize=17, loc = 'left')

    else:
        EEorEB = "_".join(name.split("_")[4:5])
        datasetname = "_".join(name.split("_")[:4])
        plt.title(f"{datasetname}",fontsize=23, loc = 'left')
    # plot the histograms
    plt.hist(center_pt, bins=binning_pt, weights=photon.pt_EB, histtype='step', label='EB',linewidth=3)
    plt.hist(center_pt, bins=binning_pt, weights=photon.pt_EE, histtype='step', label='EE',linewidth=3)
    # add the plot labels
    hep.cms.label(loc=1,com=13.6)
    plt.yscale('log')
    plt.xlabel('Photon_$p_T$ (GeV)')
    plt.ylabel('count')
    plt.legend()
    plt.xlim(pt_x_low-100, pt_x_high)
    plt.ylim(bottom=0,top = 1e6)

    plt.savefig(f'./plots/pt/{os.path.splitext(fileName)[0]}.png')
    plt.close()
    # plt.show()

# eta plot
for fileName in parquetFiles:

    file_path = os.path.join(parquetDir, fileName)
    photon = ak.from_parquet(file_path)
    # %%
    # get the name of the file
    name = os.path.splitext(fileName)[0]
    part = name.split("_")
    if len(part) > 7:
        EEorEB = "_".join(name.split("_")[5:6])
        datasetname = "_".join(name.split("_")[:5])
        plt.title(f"{datasetname}",fontsize=17, loc = 'left')

    else:
        EEorEB = "_".join(name.split("_")[4:5])
        datasetname = "_".join(name.split("_")[:4])
        plt.title(f"{datasetname}",fontsize=20, loc = 'left')
    # plot the histograms
    plt.hist(center_eta, bins=binning_eta, weights=photon.eta_EB, histtype='step', label='EB',linewidth=3)
    plt.hist(center_eta, bins=binning_eta, weights=photon.eta_EE, histtype='step', label='EE',linewidth=3)
    # add the plot labels
    hep.cms.label(loc=1,com=13.6)
    plt.yscale('log')
    plt.xlabel('photon_${\eta}$')
    plt.ylabel('count')
    plt.legend()
    plt.xlim(eta_x_low-0.25, eta_x_high+0.25)
    plt.ylim(bottom=0, top = 2*1e4)

    plt.savefig(f'./plots/eta/{os.path.splitext(fileName)[0]}.png')
    plt.close()
    # plt.show()
