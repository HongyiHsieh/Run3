import awkward as ak
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import hist
import os
from hist import Hist

hep.style.use("CMS")
palette = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]

# High pT ID cuts
pT = 125
IsoCh_value = 5
hoe_value = 0.05
r9_value = 0.8
EB_Isogamma_value = 2.75
EE_Isogamma_value = 2.0
EB_sieie_value = 0.0105
EE_sieie_value = 0.028

def calculate_binning(bin):
    width = bin[1] - bin[0]
    center = (bin[:-1] + bin[1:]) / 2
    return width, center

# Define mass binning
n_bins = 40
mass_x_low = 0
mass_x_high = 7000
binning_mass = np.linspace(mass_x_low, mass_x_high, n_bins + 1)
width_mass, center_mass = calculate_binning(binning_mass)

Path('./plots/highptId/').mkdir(exist_ok=True, parents=True)
Path('./plots/efficiency/').mkdir(exist_ok=True, parents=True)
processes = [f for f in os.listdir("./NTuples_noCut") if os.path.isdir(os.path.join("./NTuples_noCut", f))]
mass = []
efficiency = []

for process in processes:
    diphotons = ak.from_parquet(f'./NTuples_noCut/{process}/')
    print(process)

    # Select diphotons with pT > 50 GeV and electron veto
    vetoed_diphotons = diphotons[diphotons.lead_electronVeto & diphotons.sublead_electronVeto]
    selected_diphotons = vetoed_diphotons[(vetoed_diphotons.lead_pt > pT) & (vetoed_diphotons.sublead_pt > pT)]
    print(f'Veto and pT: {len(selected_diphotons) - len(diphotons)}')

    # IsoCharge 
    IsoCh = selected_diphotons[(selected_diphotons.lead_pfChargedIso < IsoCh_value) & (selected_diphotons.sublead_pfChargedIso < IsoCh_value)]
    print(f'Isocharge: {len(IsoCh) - len(selected_diphotons)}')

    # H/E
    hoe =  IsoCh[(IsoCh.lead_hoe < hoe_value) & (IsoCh.sublead_hoe < hoe_value)]
    print(f'H/E: {len(hoe) - len(IsoCh)}')

    # R9
    r9 = hoe[(hoe.lead_r9 > r9_value) & (hoe.sublead_r9 > r9_value)]
    print(f'R9: {len(r9) - len(hoe)}')

    # Isogamma
    lead_Isogamma = r9[
    (r9.lead_isScEtaEB & (r9.lead_pfPhoIso03 < EB_Isogamma_value)) | 
    (r9.lead_isScEtaEE & (r9.lead_pfPhoIso03 < EE_Isogamma_value))
    ]
    Isogamma = lead_Isogamma[
    (lead_Isogamma.sublead_isScEtaEB & (lead_Isogamma.sublead_pfPhoIso03 < EB_Isogamma_value)) | 
    (lead_Isogamma.sublead_isScEtaEE & (lead_Isogamma.sublead_pfPhoIso03 < EE_Isogamma_value))
    ]
    print(f'Isogamma: {len(Isogamma) - len(r9)}')

    # SIEIE
    lead_sieie = Isogamma[
    (Isogamma.lead_isScEtaEB & (Isogamma.lead_sieie < EB_sieie_value)) | 
    (Isogamma.lead_isScEtaEE & (Isogamma.lead_sieie < EE_sieie_value))
    ]
    sieie = lead_sieie[
    (lead_sieie.sublead_isScEtaEB & (lead_sieie.sublead_sieie < EB_sieie_value)) | 
    (lead_sieie.sublead_isScEtaEE & (lead_sieie.sublead_sieie < EE_sieie_value))
    ]
    print(f'SIEIE: {len(sieie) - len(Isogamma)}')

    # without_EEEE = sieie[sieie.lead_isScEtaEB]

    # hiPtId_mass_his = np.histogram(selected_diphotons.mass, bins=binning_mass)[0]
    hiPtId_mass_his = np.histogram(sieie.mass, bins=binning_mass)[0]
    mass_his = np.histogram(diphotons.mass, bins=binning_mass)[0]
  
    plt.hist(center_mass, bins=binning_mass, weights=hiPtId_mass_his, histtype='step',density = True, label='pass high pT ID', linewidth=3)
    plt.hist(center_mass, bins=binning_mass, weights=mass_his, histtype='step',density = True, label='no cut', linewidth=3)

    # print(np.sum(mass_his * width_mass))
    # print(hiPtId_mass_his[20:23],"\n")
    # print(mass_his[20:23],"\n")
    print("efficiency :", np.sum(hiPtId_mass_his) / np.sum(mass_his))
    efficiency.append((np.sum(hiPtId_mass_his) / np.sum(mass_his)))
    mass.append(int(process.split("_M-")[-1]))

    hep.cms.label(loc=1,com=13.6)
    plt.yscale('log')
    plt.legend()
    plt.xlabel('$M_{\gamma\gamma}$')
    plt.ylabel('Density')
    plt.title(f'{process}',fontsize=20, loc = 'left', pad= 35)
    plt.xlim(mass_x_low, mass_x_high)
    plt.ylim(bottom=0, top = 1) 
    plt.savefig(f'./plots/highptId/{process}.png')
    plt.close()

print(mass)
print(efficiency)
hep.cms.label(loc=1,com=13.6)
plt.plot(mass, efficiency, marker = 'o', linestyle = '--', label='efficiencyy')
plt.xlabel('$M_{\gamma\gamma}$')
plt.ylabel('efficiency')
plt.title('High pT ID cut efficiency',fontsize=30, loc = 'center', pad= 35)
plt.legend()
plt.ylim(bottom=0, top = 1)
# plt.xlim(800, 5500)
plt.xticks([1000, 2500, 5000], ['1000','2500', '5000'])
plt.savefig('./plots/efficiency/efficiency.png')
plt.close()