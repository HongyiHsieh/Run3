# %%
import awkward as ak
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import hist
from hist import Hist
import os

hep.style.use("CMS")
palette = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]

# %%
parquetDir = "./parquet/"

parquetFiles = [f for f in os.listdir(parquetDir) if f.endswith('.parquet')]

# %%
for fileName in parquetFiles:
    # directory for output plots
    plots_dir = './plots'
    Path(plots_dir).mkdir(exist_ok=True)

    file_path = os.path.join(parquetDir, fileName)
    photon = ak.from_parquet(file_path)

    # Define the binning
    n_bins = 40
    x_low = 0
    x_high = 5000
    binning = np.linspace(x_low, x_high, n_bins + 1)
    width = binning[1] - binning[0]
    center = (binning[:-1] + binning[1:]) / 2

    # %%
    len(center)

    # %%
    plt.hist(center, bins=binning, weights=photon.pt, histtype='step', label='EB')

    hep.cms.label("Preliminary",loc=1,com=13)
    plt.yscale('log')
    plt.xlabel('Photon_$p_T$ (GeV)')
    plt.ylabel('count')
    plt.legend()
    plt.xlim(x_low-100, x_high)
    plt.ylim(bottom=0,top = 1e6)
    name = os.path.splitext(fileName)[0]
    datasetname = "_".join(name.split("_")[:4])

    plt.title(f"{datasetname}",fontsize=20, loc = 'left')
    plt.savefig(f'./plots/{os.path.splitext(fileName)[0]}.png')
    plt.close()
    # plt.show()


