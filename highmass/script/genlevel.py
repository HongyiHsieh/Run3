import numpy as np
import json
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor
import hist
from hist import Hist
import mplhep as hep
import matplotlib.pyplot as plt
import os

class MyProcessor(processor.ProcessorABC):
    def __init__(self):
        pass
    def process(self, events):
        dataset = events.metadata['dataset']
        result ={}
        result[dataset]={
            "counts" : len(events)
        }

        costheta_hist = (
            Hist.new.StrCat([], growth=True, name="dataset", label="Primary dataset")
            .Reg(40, -1, 1, overflow=False, underflow=False, name="x", label = r"m$_{\gamma \gamma}$ [GeV]")
            .Weight()
            )

        mass_hist = (
            Hist.new.StrCat([], growth=True, name="dataset", label="Primary dataset")
            .Reg(40, 0, 7000, overflow=False, underflow=False, name="x", label = r"m$_{\gamma \gamma}$ [GeV]")
            .Weight()
            )
        

        GenPart = events.GenPart
        if 5100039 in GenPart.pdgId:
            genPhotons = GenPart[(GenPart.pdgId == 22) & (GenPart.pdgId[GenPart.genPartIdxMother] == 5100039)]
        else:
            genPhotons = GenPart[(GenPart.pdgId == 22) & (GenPart.pdgId[GenPart.genPartIdxMother] == 25)]

        countNumberOfgenPhotons = ak.num(genPhotons, axis=1)

        genPhotonsMasked = genPhotons[countNumberOfgenPhotons > 1]

        genPhotonsMasked["charge"] = ak.zeros_like(genPhotonsMasked.pt)
        
        # genPhotonsMasked_zip = ak.zip(
        #     {
        #         "pt": genPhotonsMasked.pt,
        #         "eta": genPhotonsMasked.eta,
        #         "phi": genPhotonsMasked.phi,
        #         "mass": genPhotonsMasked.mass,
        #         "charge": genPhotonsMasked.charge,
        #     },
        #     with_name="PtEtaPhiMCandidate",
        #     behavior=ak.behavior,
        # )

        genDiPhoton_pair = ak.combinations(genPhotonsMasked, 2, fields=["lead", "sublead"])
        genDiphotons = genDiPhoton_pair["lead"] + genDiPhoton_pair["sublead"]

        genDiphoton_selected = genDiphotons[genDiphotons.mass[:,0] > 0]

        genDiphoton_selected["charge"] = ak.zeros_like(genDiphoton_selected.pt)

        # genDiphoton_selected_zip = ak.zip(
        #     {
        #         "pt": genDiphoton_selected.pt,
        #         "eta": genDiphoton_selected.eta,
        #         "phi": genDiphoton_selected.phi,
        #         "mass": genDiphoton_selected.mass,
        #         "charge": genDiphoton_selected.charge,
        #     },
        #     with_name="PtEtaPhiMCandidate",
        #     behavior=ak.behavior,
        # )

        genDiphoton_selected = ak.firsts(genDiphoton_selected)

        genPhoton1 = genPhotonsMasked[:,0].boost(-genDiphoton_selected.boostvec)

        genDiphoton_cosThetaStar = np.cos(genPhoton1.theta)
        
        costheta_hist.fill(dataset=dataset, x=genDiphoton_cosThetaStar)
        #gen diphoton mass
        if 5100039 in GenPart.pdgId:
            genMother = GenPart[(GenPart.pdgId == 5100039) & (GenPart.pdgId[GenPart.genPartIdxMother] == 5100039)]
        else:
            genMother = GenPart[(GenPart.pdgId == 25) & (GenPart.pdgId[GenPart.genPartIdxMother] == 25)]
        
        mass_hist.fill(dataset=dataset, x=genMother.mass)

        result["cosTheta"] = costheta_hist
        result["mass"] = mass_hist

        return result
    
    def postprocess(self, accumulator):
        pass

# for f in os.listdir("."):
#     if f.endswith('.json'):
#         with open(os.path.join(".", f)) as file:
#             sample_dict = json.load(file)

run = processor.Runner(
    executor = processor.FuturesExecutor(workers = 4),
    schema= NanoAODSchema
)
with open("./2022_0p014.json") as file:
    sample_dict = json.load(file)
result = run(
    sample_dict,
    treename="Events",
    processor_instance=MyProcessor()
)
# print(result)
hep.style.use(hep.style.CMS)
from pathlib import Path
Path('./plots/genthetastar/').mkdir(exist_ok=True, parents=True)
Path('./plots/genMass/').mkdir(exist_ok=True, parents=True)


for key in sample_dict:
    f, ax = plt.subplots(figsize=(10,10))
    result["cosTheta"][{"dataset":key}].plot(ax=ax, density=True, linewidth=3)
    hep.cms.label("Preliminary",loc=1,com=13.6)
    plt.xlabel(r'$\cos{\theta^*}$')
    plt.ylabel('Density')
    # plt.yscale('log')
    plt.title(f'{key}',fontsize=20, loc = 'left', pad= 35)
    # plt.xlim(costhetastar_x_low-0.1, costhetastar_x_high+0.1)
    plt.ylim(bottom=0, top = 1)
    plt.savefig(f'./plots/genthetastar/{key}.png')
    plt.close()

for key in sample_dict:
    f, ax = plt.subplots(figsize=(10,10))
    result["mass"][{"dataset":key}].plot(ax=ax, density=True, linewidth=3)
    hep.cms.label("Preliminary",loc=1,com=13.6)
    plt.xlabel(r'$m_{\gamma \gamma}$ [GeV]')
    plt.ylabel('Density')
    plt.yscale('log')
    plt.title(f'{key}',fontsize=20, loc = 'left', pad= 35)
    plt.ylim(bottom=0, top = 1)
    plt.savefig(f'./plots/genMass/{key}.png')
    plt.close()