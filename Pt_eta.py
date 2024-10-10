# %%
import numpy as np
import json
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor
import hist
from pathlib import Path
from hist import Hist
import pandas as pd
import coffea
import argparse
import os


# %%
class MyProcessor(processor.ProcessorABC):
    def __init__(self):
        pass
    def process(self, events):
        dataset = events.metadata["dataset"]

        results = {}
        results[dataset] = {
            "count": len(events)
        }

        h_photon_EB = (
            Hist.new.StrCat([], growth=True, name="dataset_photon_EB_pt", label="Dataset")
            .Reg(40, 0, 5000, overflow=False, underflow=False, name="x", label = "[Gev]" )
            .Weight()
        )
        h_photon_EE = (
            Hist.new.StrCat([], growth=True, name="dataset_photon_EE_pt", label="Dataset")
            .Reg(40, 0, 5000, overflow=False, underflow=False, name="x", label = "[Gev]" )
            .Weight()
        )
        # separate the events into EB and EE
        photon_EB = events.Photon[events.Photon.isScEtaEB]
        photon_EE = events.Photon[events.Photon.isScEtaEE]

        photon_EB_no_none = photon_EB[~ak.is_none(photon_EB, axis=1)]      
        photon_EE_no_none = photon_EE[~ak.is_none(photon_EE, axis=1)]      
        
        # EB_total_selection = ak.fill_none(
        #     ak.num(photon_EB_selected,axis=1) > 1,
        #     False
        # )

        # EE_total_selection = ak.fill_none(
        #     ak.num(photon_EE,axis=1) > 1,
        #     False
        # )

        EB_photon_selection = photon_EB_no_none[ak.num(photon_EB_no_none,axis=1) > 1]
        EE_photon_selection = photon_EE_no_none[ak.num(photon_EE_no_none,axis=1) > 1]

        EB_photon_selection = EB_photon_selection[:,:2]
        EE_photon_selection = EE_photon_selection[:,:2]

        h_photon_EB.fill(dataset_photon_EB_pt=dataset, x=ak.flatten(EB_photon_selection.pt))
        h_photon_EE.fill(dataset_photon_EE_pt=dataset, x=ak.flatten(EE_photon_selection.pt))

        results["photon_EB_pt"] = h_photon_EB
        results["photon_EE_pt"] = h_photon_EE
        # results["photon_EB_pt"] = photon_EB_selection
        # results["photon_EE_pt"] = photon_EE_selection

        return results
    
    def postprocess(self, accumulator):
        pass


# %%
# samplejson = "/eos/home-h/hhsieh/Run3/mc/2022/GGspin0/0p014/"
for f in os.listdir("."):
    if f.endswith('.json'):
        with open(os.path.join(".", f)) as file:
            sample_dict = json.load(file)

# %%
run = processor.Runner(
    # executor=processor.IterativeExecutor(),
    executor=processor.FuturesExecutor(workers=10), # user 4 cores
    schema=NanoAODSchema
)

results = run(
    sample_dict,
    treename="Events",
    processor_instance=MyProcessor(),
)

# %%
# cell 23

# import mplhep as hep
# import matplotlib.pyplot as plt
for key in sample_dict :
    for EBorEE in ["EB","EE"]:
        plots_dir = './parquet'
        Path(plots_dir).mkdir(exist_ok=True)
        # hep.style.use(hep.style.CMS)

        # f, ax = plt.subplots(figsize=(10,10))

        # ax.set_ylabel("Count")
            
        # results[f"photon_{EBorEE}_pt"][{f"dataset_photon_{EBorEE}_pt":key}].plot(ax=ax, label=f'MC_{EBorEE}')

        value = results[f"photon_{EBorEE}_pt"][{f"dataset_photon_{EBorEE}_pt":key}].values()
        edges = results[f"photon_{EBorEE}_pt"][{f"dataset_photon_{EBorEE}_pt":key}].axes.edges
        center = [(edges[0][i] + edges[0][i+1]) / 2 for i in range(len(edges[0])-1)]

        pt_data={
                # 'pt':[np.repeat(center, value[i]) for i in range(len(value))]
                'pt':value
        }

        df_pt = pd.DataFrame(data=pt_data)
        df_pt.to_parquet(f'./parquet/{key}_{EBorEE}_photon_pt.parquet')

        # hep.cms.label(loc=2,com=13.6)
    
        # plt.yscale('log')
        # plt.xlabel('$p_T$ (GeV)')
        # plt.title(key,fontsize=20, loc = 'left')
        # plt.legend()
        # plt.savefig(f"plots/{key}_photon_{EBorEE}_pt.png")
        


